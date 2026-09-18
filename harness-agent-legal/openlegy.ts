/**
 * pi extension — OpenLegy MCP client.
 *
 * Exposes OpenLegy (Légifrance) MCP tools to a pi agent as native pi tools:
 *  - openlegy_search_code        : rechercher_code (Légifrance)
 *  - openlegy_get_article        : rechercher_code champ=NUM_ARTICLE
 *  - openlegy_search_jurisprudence : rechercher_jurisprudence_judiciaire
 *  - openlegy_get_jurisprudence  : get_decision_judiciaire (JURITEXT)
 *  - openlegy_list_codes         : lister_codes_juridiques
 *
 * Token read from env OPENLEGY_TOKEN.
 *
 * The MCP endpoint is streamable-HTTP. We open a fresh session per tool call to
 * avoid stale-state issues; the server returns an mcp-session-id that we honour.
 */
import { Type } from "typebox";
import fs from "node:fs";

const ENDPOINT = process.env.OPENLEGY_ENDPOINT ?? "https://mcp.openlegi.fr/legifrance/mcp";
const TOKEN = process.env.OPENLEGY_TOKEN ?? "";
const PROJECT = "harness-agent-legal";
const BASE_DIR = new URL(".", import.meta.url).pathname;

let LOG_PATH = `${BASE_DIR}openlegy-calls.log`;

function appendLog(line: string) {
  try {
    fs.appendFileSync(LOG_PATH, `${new Date().toISOString()} ${line}\n`);
  } catch {
    /* best effort */
  }
}

async function mcpCall(name: string, args: Record<string, unknown>): Promise<string> {
  if (!TOKEN) {
    throw new Error("OPENLEGY_TOKEN env var is not set — cannot call OpenLegy MCP.");
  }
  const endpoint = ENDPOINT;
  // 1) initialize a fresh session
  const init = await rawMcp(endpoint, null, {
    jsonrpc: "2.0",
    id: 1,
    method: "initialize",
    params: {
      protocolVersion: "2025-03-26",
      capabilities: {},
      clientInfo: { name: "pi-openlegy", version: "0.1" },
    },
  });
  const sessionId = init.sessionId;
  // 2) send initialized notification
  await rawMcp(endpoint, sessionId, {
    jsonrpc: "2.0",
    method: "notifications/initialized",
    params: {},
  });
  // 3) call the tool
  const call = await rawMcp(endpoint, sessionId, {
    jsonrpc: "2.0",
    id: 2,
    method: "tools/call",
    params: { name, arguments: args },
  });
  const body = call.body;
  if (body.jsonrpc && body.id === 1) {
    // it's the init response captured wrong; fall through
  }
  if (body.error) {
    throw new Error(`OpenLegy MCP error: ${JSON.stringify(body.error)}`);
  }
  const content = body.result?.content;
  if (!Array.isArray(content)) {
    throw new Error(`OpenLegy MCP returned no content: ${JSON.stringify(body)}`);
  }
  const text = content
    .filter((c: any) => c?.type === "text")
    .map((c: any) => c.text)
    .join("\n");
  return text;
}

interface RawResponse {
  sessionId: string | null;
  body: any;
}

async function rawMcp(
  endpoint: string,
  sessionId: string | null,
  payload: Record<string, unknown>,
): Promise<RawResponse> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    Accept: "application/json, text/event-stream",
    Authorization: `Bearer ${TOKEN}`,
  };
  if (sessionId) headers["mcp-session-id"] = sessionId;
  const resp = await fetch(endpoint, {
    method: "POST",
    headers,
    body: JSON.stringify(payload),
  });
  if (!resp.ok) {
    throw new Error(`OpenLegy HTTP ${resp.status}: ${await resp.text()}`);
  }
  const text = await resp.text();
  let session = resp.headers.get("mcp-session-id") || null;
  const body = parseSseOrJson(text);
  return { sessionId: session, body };
}

function parseSseOrJson(text: string): any {
  const lines = text.split("\n");
  let dataLine = "";
  for (const line of lines) {
    if (line.startsWith("data:")) {
      dataLine += line.slice(5).trim();
    } else if (line.startsWith("{")) {
      // possibly coalesced JSON on one line
      dataLine += line.trim();
    }
  }
  if (dataLine) {
    try {
      return JSON.parse(dataLine);
    } catch {
      return { raw: text };
    }
  }
  try {
    return JSON.parse(text);
  } catch {
    return { raw: text };
  }
}

export default function (pi: any) {
  const tools: Array<{
    name: string;
    label: string;
    description: string;
    params: any;
    rpc: string;
    map: (args: any) => Record<string, unknown>;
  }> = [
    {
      name: "openlegy_search_code",
      label: "OpenLegy: rechercher des articles dans un code",
      description:
        "Recherche des articles dans les codes juridiques français (Légifrance) — par mots-clés, dans tout le code ou dans un champ. Utilisez code_name exact (ex. 'Code civil', 'Code pénal', 'Code du travail'). Fournit texte intégral + numéro + état juridique + lien Légifrance de chaque article trouvé.",
      params: Type.Object({
        search: Type.String({ description: "Termes de recherche (ex: 'légitime défense', 'contrat de travail')" }),
        code_name: Type.String({ description: "Nom exact du code juridique (ex: 'Code pénal', 'Code civil'). Utilisez openlegy_list_codes pour lister." }),
        champ: Type.Optional(Type.String({ description: "Champ: ALL, TITLE, TABLE, NUM_ARTICLE, ARTICLE. Défaut ALL." })),
        page_size: Type.Optional(Type.Number({ description: "Nombre de résultats (défaut 7, max 100)" })),
      }),
      rpc: "rechercher_code",
      map: (a) => ({ search: a.search, code_name: a.code_name, champ: a.champ ?? "ALL", page_size: Math.min(a.page_size ?? 7, 10) }),
    },
    {
      name: "openlegy_get_article",
      label: "OpenLegy: obtenir un article précis d'un code",
      description:
        "Récupère le texte intégral d'un article précis d'un code juridique. Indiquez le numéro d'article (ex '122-5', 'L1237-11') et le nom du code. Renvoie le texte + état juridique + lien Légifrance.",
      params: Type.Object({
        article_number: Type.String({ description: "Numéro de l'article (ex: '122-5', '1240', 'L1237-11')" }),
        code_name: Type.String({ description: "Nom exact du code juridique (ex: 'Code pénal', 'Code civil')" }),
      }),
      rpc: "rechercher_code",
      map: (a) => ({ search: a.article_number, code_name: a.code_name, champ: "NUM_ARTICLE" }),
    },
    {
      name: "openlegy_search_jurisprudence",
      label: "OpenLegy: rechercher des jurisprudences judiciaires",
      description:
        "Recherche dans la jurisprudence judiciaire française (Cour de cassation, cours d'appel, tribunaux). Par mots-clés, éventuellement filtré par juridiction et date. Renvoie décisions + identifiants JURITEXT + liens Légifrance. Pour le texte intégral d'une décision, utilisez openlegy_get_jurisprudence avec le JURITEXT.",
      params: Type.Object({
        search: Type.String({ description: "Termes de recherche (ex: 'légitime défense', 'abus de droit')" }),
        juridiction_judiciaire: Type.Optional(Type.String({ description: "Juridiction ou liste JSON: 'Cour de cassation', 'Juridictions d'appel', 'Juridictions du premier degré'" })),
        date_debut: Type.Optional(Type.String({ description: "Date début YYYY-MM-DD" })),
        date_fin: Type.Optional(Type.String({ description: "Date fin YYYY-MM-DD" })),
        page_size: Type.Optional(Type.Number({ description: "Nombre de résultats (défaut 7, max 100)" })),
        panorama: Type.Optional(Type.Boolean({ description: "Vrai = seulement métadonnées + résumés, plus compact" })),
      }),
      rpc: "rechercher_jurisprudence_judiciaire",
      map: (a) => {
        const base: Record<string, unknown> = {
          search: a.search,
          page_size: Math.min(a.page_size ?? 7, 10),
        };
        if (a.panorama !== undefined) base.panorama = a.panorama;
        if (a.date_debut) base.date_debut = a.date_debut;
        if (a.date_fin) base.date_fin = a.date_fin;
        if (a.juridiction_judiciaire) {
          const raw = a.juridiction_judiciaire;
          try {
            base.juridiction_judiciaire = JSON.parse(raw);
          } catch {
            base.juridiction_judiciaire = [String(raw)];
          }
        }
        return base;
      },
    },
    {
      name: "openlegy_get_jurisprudence",
      label: "OpenLegy: texte intégral d'une décision judiciaire",
      description:
        "Récupère le texte intégral d'une décision de jurisprudence judiciaire à partir de son identifiant Légifrance JURITEXT (ex: 'JURITEXT000047577096'). Se combine avec openlegy_search_jurisprudence.",
      params: Type.Object({
        identifiant: Type.String({ description: "Identifiant Légifrance JURITEXT de la décision" }),
      }),
      rpc: "get_decision_judiciaire",
      map: (a) => ({ identifiant: a.identifiant }),
    },
    {
      name: "openlegy_list_codes",
      label: "OpenLegy: lister les codes juridiques disponibles",
      description: "Liste tous les codes juridiques disponibles sur Légifrance. À utiliser pour connaître les code_name exacts avant une recherche.",
      params: Type.Object({}),
      rpc: "lister_codes_juridiques",
      map: () => ({}),
    },
  ];

  for (const tool of tools) {
    pi.registerTool({
      name: tool.name,
      label: tool.label,
      description: tool.description,
      parameters: tool.params,
      async execute(_toolCallId, args, _signal, _onUpdate) {
        const started = Date.now();
        let ok = true;
        let summary = "";
        try {
          const result = await mcpCall(tool.rpc, tool.map(args));
          summary = `${tool.rpc} -> ${result.slice(0, 120)}`;
          return { content: [{ type: "text", text: result }], details: {} };
        } catch (err: any) {
          ok = false;
          summary = `${tool.rpc} ERROR: ${err?.message ?? err}`;
          throw err;
        } finally {
          appendLog(
            `${JSON.stringify({ project: PROJECT, tool: tool.rpc, name: tool.name, ms: Date.now() - started, ok, summary: summary.slice(0, 200) })}`,
          );
        }
      },
    });
  }
}