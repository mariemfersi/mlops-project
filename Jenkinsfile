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
                sh '''
                echo "🐍 Checking Python installation..."
                which python3 || echo "Python not found"
                python3 --version
                '''
            }
        }

        stage('Setup Environment') {
    steps {
        sh '''
        echo "⚙️ Setting up virtual environment..."
        python3 -m venv $VENV || { echo "❌ Failed to create venv. Check python3-venv installation"; exit 1; }
        . $VENV/bin/activate
        $PIP install --upgrade pip
        $PIP install -r requirements.txt
        echo "✅ Virtual environment ready."
        '''
    }
}


        stage('Prepare Data') {
            steps {
                sh '''
                echo "📦 Preparing data..."
                . $VENV/bin/activate
                $PYTHON main.py --prepare --data_path $DATA_PATH --target $TARGET
                echo "✅ Données préparées."
                '''
            }
        }

        stage('Train Model') {
            steps {
                sh '''
                echo "🤖 Training model..."
                . $VENV/bin/activate
                $PYTHON main.py --train --prepare --data_path $DATA_PATH --target $TARGET
                echo "✅ Modèle entraîné."
                '''
            }
        }

        stage('Evaluate Model') {
            steps {
                sh '''
                echo "🧮 Evaluating model..."
                . $VENV/bin/activate
                $PYTHON main.py --evaluate --prepare --train --data_path $DATA_PATH --target $TARGET
                echo "✅ Évaluation terminée."
                '''
            }
        }

        stage('Save Model') {
            steps {
                sh '''
                echo "💾 Saving model..."
                . $VENV/bin/activate
                $PYTHON main.py --save --prepare --train --data_path $DATA_PATH --target $TARGET --model_path $MODEL_PATH
                echo "✅ Modèle sauvegardé dans $MODEL_PATH."
                '''
            }
        }

        stage('Lint & Format') {
            steps {
                echo "Skipping flake8 linting"
            }
        }


        stage('Test Environment') {
            steps {
                sh '''
                echo "🧪 Running environment tests..."
                . $VENV/bin/activate
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
            echo "🧹 Nettoyage en cours..."
            rm -rf __pycache__ *.pyc *.pyo *.png
            echo "✅ Nettoyage terminé."
            '''
        }
    }
}
