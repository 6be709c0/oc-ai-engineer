J'ai un projet d'école.

```
Fashion-Insta est une entreprise du monde de la mode qui commercialise des articles vestimentaires. Elle dispose d’un réseau de magasins physiques, mais également d’un site e-commerce qui lui permet de commercialiser ses produits selon ces deux canaux.

 
Vous êtes IA product manager et vous travaillez dans l’équipe d’Alicia, VP product, pour mener à bien des projets IA au sein de l’entreprise. 

Alicia est chargée de construire et d’exécuter la vision produit de l’entreprise. Dans sa roadmap de projets prévus, le projet IA identifié comme ayant le plus de potentiel est le développement d’une application mobile de recommandation d’articles vestimentaires basée sur des photos.

L’objectif de l’application mobile est de permettre aux utilisateurs de l’application de se prendre en photo avec leurs habits favoris et d’obtenir en retour des recommandations d’articles du même style vestimentaire.

L’application sera réalisée en s’appuyant sur les outils cloud fournis par Microsoft Azure, le partenaire cloud de “Fashion-Insta”.
```

Voici ce que j'ai comme équipe

- Product Manager
- Frontend Dev Mobile
- Backend-Dev
- Devops Engineer	
- Data Scientist Junior	
- Sous traitant	
- AI Engineer	
- QA Engineer

Voici ce que j'ai comme sprint
```
TASK NAME,STORY POINTS,START DATE,"END 
DATE"
SPRINT 1,,08/05,08/16
Inscription via une adresse mail,8,08/05,08/06
Connexion via une adresse mail,5,08/07,08/07
Reset du mot de passe via une adresse mail,3,08/08,08/08
Se prendre en photo,2,08/09,08/09
Supprimer une photo,1,08/12,08/12
Gérer des collection de photos,8,08/13,08/14
"Gérer ses préférence (styles, marques, sites)",5,08/15,08/16
Reconnaissance de vêtements à partir de photo,13,08/05,08/14
SPRINT 2,,08/19,08/30
Mettre les produits dans un panier,,08/19,08/21
Valider une commande,,08/22,08/26
Choisir un type de livraison,,08/27,08/28
Effectuer une transaction bancaire,,08/28,08/30
Recommandation de vêtements depuis ses préférences,,08/19,08/28
SPRINT 3,,09/02,09/13
Recommandation de vêtements depuis sa garde-robe,,09/04,09/13
Proposer plusieurs vêtements sur une photo,,09/02,09/06
Changer les caractéristiques du vêtement proposé,,09/09,09/11
Laisser un avis sur les vêtements proposés,,09/12,09/13
SPRINT 4,,09/16,09/27
Accéder à la modification et suppression des données personnelles,,09/16,09/17
Gérer la durée de conservation des données personnelles,,09/18,09/19
Gérer la désinscription au service de recommandation,,09/20,09/23
Réaliser une purge automatique des données personnelles au bout d’un délai maximum fixé sans activité de l’utilisateur,,09/24,09/25
```

Voici mes facteurs de risque:
```
Constats (facteurs de risque) :
● Le Data Scientist est junior, il s’appuie sur les compétences d’un sous-traitant pour réaliser les
modèles
● Les délais sont très courts car un concurrent planifie le même type d’application, il faut être
les premiers à proposer ces services
● Les développeurs de l’application mobile travaillent en parallèle sur une autre application
urgente
● Les données de type image sont des données personnelles sensibles
● La gestion des données personnelles et leur sécurité est un enjeu majeur, pour rassurer les
clients de poster leurs photos et leurs préférences
```

Je faire une analyse complète des risques concernant la réalisation du projet, en plus de ceux concernant la gestion des données personnelles évoquée ci-dessus : détection des risques, conséquences, priorisation et plan d’action pour maîtriser ces risques. 
Pour ce faire, on me donne ce CSV en exemple:
```
Facteurs de risque,"Risque
(événement redouté)",Conséquence,Impact,"Conséquences
(en coût, délai, qualité, satisfaction client)","Impact
(0 à3)","Probabilité
(0 à 3)","Criticité
(impact*prob)","Actions de prévention
(pour éviter l'événement redouté)","Action de correction
(si événement redouté avéré)"
Etant donné que …,Si …,,,Alors …,,,,,
Seules 2 ressources critiques connaissent le logiciel,Non disponibilité ou perte de la compétence,Rétro-ingénierie ou rework importante,Activité de rétro-ingénierie des règles à re-documenter => charge + formation + délai,"Certaines activités ne pourront être réalisées.
Il faudra former des compétences juniors pour remplacer le manque.

Impact = coût, délai et qualité",2,3,6,". Affecter des ressources pour faire le transfert de compétence
",". Recadrer le projet
. Maintenir l'ancien système
. Lancer une action de rétro-ingénierie des règles et re-documenter"
,,,,,0,0,0,,
,,,,,0,0,0,,
,,,,,0,0,0,,
,,,,,0,0,0,,
,,,,,0,0,0,,
,,,,,0,0,0,,
```

Donne moi un nouveau CSV pour ce projet