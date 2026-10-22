
# Comparatif des approches — projet fraude (TP04)

Date de consultation des documentations cloud : 22/10/2026.

| Approche | F1 test | Temps humain | Temps machine | Coût estimé (source, date) | Explicabilité | Export / réversibilité |
|---|---:|---|---|---|---|---|
| Logistic Regression (référence) | 0.7016 | Faible, réglage simple | 0.09 s | 0 USD de calcul local, hors électricité | Bonne | Totale, modèle local |
| Random Forest (référence) | 0.7893 | Faible à moyen | 3.31 s | 0 USD de calcul local, hors électricité | Moyenne | Totale, modèle local |
| Réglage manuel (TP2) | 0.8669 | Moyen à élevé, réglage manuel | Non relevé | 0 USD de calcul local, hors électricité | Dépend du modèle | Bonne si format standard |
| FLAML (60 s) | 0.8075 | Faible à moyen, préparation et contrôle | 59.95 s | 0 USD de calcul local, hors électricité | Moyenne à faible selon le modèle | Bonne, modèle exportable |
| Vertex AI AutoML tabulaire | Non mesuré | Faible à moyen | Non mesuré | Variable : budget d'entraînement et node-hours, plus ressources de déploiement. Source : Google Cloud Pricing, consultée le 22/10/2026 | Bonne, explications disponibles selon le modèle et le mode de service | Export possible pour classification/régression tabulaire, avec limitations |
| SageMaker Autopilot | Non mesuré | Faible à moyen | Non mesuré | Variable : calcul, stockage et ressources d'inférence utilisés. Source : AWS SageMaker Pricing, consultée le 22/10/2026 | Bonne : importance des variables, rapports et notebooks générés | Partielle à bonne : artefacts exportables, dépendances à vérifier |
| Azure Automated ML | Non mesuré | Faible à moyen | Non mesuré | Variable : machines de calcul, stockage et déploiement. Source : Azure Machine Learning Pricing, consultée le 22/10/2026 | Bonne : métriques, graphiques et outils d'explicabilité | Bonne, artefact du modèle exportable avec ses dépendances |

## Commentaire (5 lignes)

- Le réglage manuel est pertinent lorsque l'on veut contrôler précisément les choix techniques et optimiser les performances.
- FLAML convient aux expérimentations locales rapides, avec un budget de calcul maîtrisé et une bonne intégration Python.
- Vertex AI est adapté aux équipes utilisant Google Cloud qui veulent centraliser entraînement, export et déploiement.
- SageMaker Autopilot convient aux projets intégrés à AWS nécessitant des essais documentés et des rapports d'explicabilité.
- Azure Automated ML est intéressant pour les organisations utilisant déjà Azure et souhaitant gérer expériences, modèles et endpoints dans un même environnement.

## Sources officielles consultées

- Google Vertex AI — entraînement tabulaire :
  https://docs.cloud.google.com/vertex-ai/docs/tutorials/tabular-automl/dataset-train
- Google Vertex AI — export des modèles tabulaires :
  https://docs.cloud.google.com/gemini-enterprise-agent-platform/machine-learning/export/export-model-tabular
- Google Cloud — tarification :
  https://cloud.google.com/vertex-ai/pricing
- AWS — SageMaker Autopilot :
  https://docs.aws.amazon.com/sagemaker/latest/dg/autopilot-automate-model-development.html
- AWS — tarification :
  https://aws.amazon.com/sagemaker/ai/pricing/
- Microsoft Learn — résultats AutoML :
  https://learn.microsoft.com/en-us/azure/machine-learning/how-to-understand-automated-ml
- Microsoft Learn — explicabilité :
  https://learn.microsoft.com/en-us/azure/machine-learning/how-to-machine-learning-interpretability-automl
- Azure — tarification :
  https://azure.microsoft.com/en-us/pricing/details/machine-learning/
