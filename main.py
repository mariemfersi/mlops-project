import argparse
from model_pipeline import prepare_data, select_and_train_model, evaluate_model, save_model


def main(args):
    if args.prepare:
        X_train, X_test, y_train, y_test = prepare_data(args.data_path, args.target)
        print("Données préparées.")

    if args.train:
        model = select_and_train_model(X_train, y_train)
        print("Modèle entraîné.")

    if args.evaluate:
        metrics = evaluate_model(model, X_test, y_test)
        print(f"Évaluation du modèle : {metrics}")

    if args.save:
        save_model(model, args.model_path)
        print(f"Modèle sauvegardé dans {args.model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline ML pour insurance.csv")

    parser.add_argument(
        "--prepare", action="store_true", help="Préparer les données"
    )
    parser.add_argument(
        "--train", action="store_true", help="Entraîner le modèle"
    )
    parser.add_argument(
        "--evaluate", action="store_true", help="Évaluer le modèle"
    )
    parser.add_argument(
        "--save", action="store_true", help="Sauvegarder le modèle"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="insurance.csv",
        help="Chemin vers le CSV"
    )
    parser.add_argument(
        "--target",
        type=str,
        default="charges",
        help="Nom de la colonne cible"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="model.pkl",
        help="Chemin du modèle sauvegardé"
    )

    args = parser.parse_args()
    main(args)
