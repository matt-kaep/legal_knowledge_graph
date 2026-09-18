/**
 * pi extension — append_reference tool.
 *
 * Permet à l'agent d'écrire chaque référence (article ou jurisprudence)
 * dans un fichier de sortie CSV dédié à la question courante, DÈS qu'il
 * l'identifie. L'écriture est append-only : rien n'est perdu si le process
 * est tué avant la fin (timeout, erreur).
 *
 * Le chemin du fichier est fourni dans la variable d'env
 * APPEND_REFS_PATH (défini par l'orchestrateur à chaque run).
 */
import { Type } from "typebox";
import fs from "node:fs";

export default function (pi: any) {
  pi.registerTool({
    name: "append_reference",
    label: "Écrire une référence dans la sortie",
    description:
      "Écrit une référence (article de loi ou jurisprudence) dans le fichier de réponses de la question courante, dans l'ordre de pertinence. À utiliser dès que tu identifies une référence. Une ligne par référence : la première ligne écrite est la référence la plus pertinente, la suivante la deuxième, etc. N'écris que des références que tu es sûr de vouloir garder dans ta réponse finale. Maximum 10 références au total pour cette question.",
    parameters: Type.Object({
      reference: Type.String({ description: "La référence exacte (ex: 'Code de procédure pénale, article 63-1' ou 'Cass. crim., 5 oct. 2016, n° 15-90.009')" }),
      ordre: Type.Number({ description: "Position de pertinence : 1 = plus pertinente, 2 = deuxième, etc. (de 1 à 10)" }),
    }),
    async execute(_toolCallId, args) {
      const path = process.env["APPEND_REFS_PATH"] || "";
      if (!path) {
        throw new Error("APPEND_REFS_PATH n'est pas défini — impossible d'écrire la sortie.");
      }
      const reference = String(args.reference ?? "").trim();
      const ordre = Number(args.ordre ?? 0);
      if (!reference) throw new Error("reference vide");
      if (ordre < 1 || ordre > 10) throw new Error("ordre doit être entre 1 et 10");
      const line = `${ordre}\t${reference}\n`;
      // append-only
      try {
        fs.appendFileSync(path, line, "utf-8");
      } catch (e) {
        throw new Error(`impossible d'écrire ${path} : ${e}`);
      }
      return {
        content: [{ type: "text", text: `Référence n°${ordre} sauvegardée (${reference})` }],
        details: { ordre, reference },
      };
    },
  });
}