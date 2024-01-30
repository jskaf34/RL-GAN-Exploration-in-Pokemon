# PROJET3A

## Environnement Pokemon

### Choix de la méthode pour KNN
J'ai choisi, la méthode implémentée plutôt que celle de l'env original parce qu'on ne peut pas modifier la mémoire avec la méthode originelle. En effet, dans le code d'origine, lorsqu'on a rempli la mémoire, on est bloqué donc dans leur code ce qui est fait c'est une réinitialisation. Je n'aimais pas trop ça du coup j'ai préféré coder le truc à la main, ou cette fois lorsqu'on est plein on remplace les plus vieux, etc... J'ai comparé les méthodes (cf resultats en dessous), la notre est évidemment bien plus lente pour le knn mais bcp plus rapide pour l'ajout d'image. En fait celle de base construit un graphe knn lors de l'ajout.

#### Résultats memory speed tests : 
100%|██████████████████████████████████████████████████████████████████████████| 20000/20000 [00:19<00:00, 1017.66it/s]  
Custom filling time : 19.674415826797485  
100%|████████████████████████████████████████████████████████████████████████████| 20000/20000 [45:23<00:00,  7.34it/s]  
Opt filling time : 2723.476750612259  
Custom search : 31.21134901046753  
Opt search : 0.3584418296813965  
[[[152.10453916]]] [[11381.666]] [[0]] [[4070]]  

### Choix des actions
J'ai décidé d'ajouter l'action "0: ne rien faire", parce que lors des cinématiques de combat ou de discussion, avec le tick de pyboy il faut attendre bcp de frames avant d'avoir vraiment une action à faire, je me suis dit que c'était mieux si l'agent pouvait apprendre à ne rien faire dans ces moments là (comme nous on ferait).

## DQN