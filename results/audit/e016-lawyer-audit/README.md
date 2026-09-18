# Paquet d’audit avocat E016

Statut : **bloqué sur intervention humaine**. Les 100 cas et leur clé privée
existent dans le checkout de données local, mais `lawyer_agreement.json` n’existe
pas encore. Le paquet Git ne contient ni textes juridiques ni clé d’identification.

## Procédure

1. Ouvrir `lawyer_audit_sample.csv` dans un environnement approuvé, sans publier les
   textes ou identifiants.
2. Pour chaque ligne, lire la question, la fiche Step1 et la JP candidate, puis saisir
   le verdict avocat prévu par le schéma E016 ; ne pas consulter la classe LLM pendant
   l’annotation.
3. Conserver la clé privée hors Git et produire `lawyer_agreement.json` avec le script
   `05-Technique/benchmark/etape1_embedding_pur/scripts/79_summarize_g7_graded_jp_lawyer_audit.py`.
4. Recalculer les SHA-256, puis relancer l’audit/export. Tant que l’accord pondéré et
   la précision A/B ne sont pas présents, E016 et E017 restent exploratoires.

La stratification attendue est A=27, B=22, C=17, D=17, E=17. Les chemins, tailles,
hashes et l’absence du fichier de sortie sont dans `manifest.json`.
