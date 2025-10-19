pipeline {
    agent any

    environment {
        VENV = "venv"
        PYTHON = "${VENV}/bin/python"
        PIP = "${VENV}/bin/pip"
        DATA_PATH = "insurance.csv"
        TARGET = "charges"
        MODEL_PATH = "model.pkl"
        API_PORT = "8000"
        NOTEBOOK_PORT = "8888"
    }

    stages {
        stage('Check Python') {
    steps {
        sh 'which python3 || echo "Python not found"'
        sh 'python3 --version'
    }
}

        stage('Setup Environment') {
            steps {
                sh '''
                if [ ! -d "$VENV" ]; then
                    python3 -m venv $VENV
                fi
                $PIP install --upgrade pip
                $PIP install -r requirements.txt
                echo "✅ Virtual environment ready."
                '''
            }
        }

        stage('Prepare Data') {
            steps {
                sh '''
                $PYTHON main.py --prepare --data_path $DATA_PATH --target $TARGET
                echo "✅ Données préparées."
                '''
            }
        }

        stage('Train Model') {
            steps {
                sh '''
                $PYTHON main.py --train --prepare --data_path $DATA_PATH --target $TARGET
                echo "✅ Modèle entraîné."
                '''
            }
        }

        stage('Evaluate Model') {
            steps {
                sh '''
                $PYTHON main.py --evaluate --prepare --train --data_path $DATA_PATH --target $TARGET
                echo "✅ Évaluation terminée."
                '''
            }
        }

        stage('Save Model') {
            steps {
                sh '''
                $PYTHON main.py --save --prepare --train --data_path $DATA_PATH --target $TARGET --model_path $MODEL_PATH
                echo "✅ Modèle sauvegardé dans $MODEL_PATH."
                '''
            }
        }

        stage('Lint & Format') {
            steps {
                sh '''
                $PYTHON -m flake8 . --exclude=$VENV,__pycache__
                $PYTHON -m black .
                echo "✅ Linting & Formatting OK."
                '''
            }
        }

        stage('Test Environment') {
            steps {
                sh '''
                $PYTHON test_environment.py
                echo "✅ Tests exécutés."
                '''
            }
        }
    }

    post {
        success {
            echo '🎉 Pipeline complet exécuté avec succès!'
        }
        failure {
            echo '❌ Une erreur est survenue dans le pipeline.'
        }
        cleanup {
            sh '''
            rm -rf __pycache__ *.pyc *.pyo *.png
            echo "🧹 Nettoyage terminé."
            '''
        }
    }
}

