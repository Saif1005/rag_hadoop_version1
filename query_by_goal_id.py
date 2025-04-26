

"""
Script pour rechercher des informations par goal_id
"""

import argparse
from retrivial_data_from_hdfs import (
    list_available_goals, 
    display_similar_chunks_by_goal_id
)

def main():
    """Point d'entrée principal du script"""
    parser = argparse.ArgumentParser(description="Recherche d'informations par goal_id")
    
    # Commandes disponibles
    subparsers = parser.add_subparsers(dest="command", help="Commande à exécuter")
    
    # Commande pour lister les goals disponibles
    list_parser = subparsers.add_parser("list", help="Lister les goals disponibles")
    
    # Commande pour rechercher par goal_id
    search_parser = subparsers.add_parser("search", help="Rechercher des informations par goal_id")
    search_parser.add_argument("--goal_id", "-g", type=str, required=True, help="Identifiant du goal")
    search_parser.add_argument("--query", "-q", type=str, required=True, help="Requête de recherche")
    search_parser.add_argument("--top_k", "-k", type=int, default=5, help="Nombre de résultats à afficher (défaut: 5)")
    
    # Traitement des arguments
    args = parser.parse_args()
    
    # Exécution de la commande
    if args.command == "list":
        # Lister les goals disponibles
        list_available_goals()
    
    elif args.command == "search":
        # Rechercher par goal_id
        display_similar_chunks_by_goal_id(
            query_text=args.query,
            goal_id=args.goal_id,
            top_k=args.top_k
        )
    
    else:
        # Commande non reconnue
        parser.print_help()

if __name__ == "__main__":
    main() 