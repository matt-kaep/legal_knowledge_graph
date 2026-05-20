#!/usr/bin/env bash
# cluster_check.sh — détecte ce qu'il faut mettre dans pilot_slurm.sh
# Lance-le sur le cluster :   bash cluster_check.sh
set -u

echo "=== 1) Partitions disponibles (GRES = ressources GPU) ==="
echo "    Cherche une partition avec 'gpu' dans la colonne GRES."
echo "    L'astérisque (*) à côté du nom = partition par défaut."
sinfo -o "%20P %10l %10D %20G %15f" 2>/dev/null || echo "  (sinfo indisponible)"

echo
echo "=== 2) Nœuds avec L40S (features) ==="
sinfo -o "%20N %20P %30f %20G" 2>/dev/null | grep -i -E "l40|h100|a100" \
  || echo "  (aucun nœud L40S/H100/A100 détecté via features — vérifie %G)"

echo
echo "=== 3) Compte SLURM ==="
sacctmgr -nP show user "$USER" format=user,defaultaccount,account 2>/dev/null \
  || echo "  (sacctmgr indisponible : ton cluster n'exige probablement pas --account)"

echo
echo "=== 4) Tes jobs en cours / récents (pour voir la partition utilisée d'habitude) ==="
squeue -u "$USER" -o "%.10i %.20P %.10T %.20j %.5D %R" 2>/dev/null || true
sacct -u "$USER" -X --format=JobID,Partition,AllocTRES%40,State -S "$(date -d '-7 days' +%F 2>/dev/null || date -v-7d +%F)" 2>/dev/null | head -10 || true

echo
echo "=== Conclusion ==="
echo " - Si la partition par défaut (astérisque en 1) propose des GPU L40S,"
echo "   tu peux laisser --partition et --constraint commentés."
echo " - Sinon, mets le nom de partition GPU vu en 1 dans pilot_slurm.sh"
echo "   (#SBATCH --partition=<nom>)."
echo " - Si en 3 'defaultaccount' est non vide ET que ton admin l'exige,"
echo "   décommente #SBATCH --account=<defaultaccount>."
echo " - Si en 2 plusieurs types de GPU coexistent, ajoute --constraint=l40s."
