import os
import json
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# NLTK for text processing
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')
class PostLoginIntentClassifier:
    def __init__(self, intents_file='data.json'):
        """
        Initialize the Post-Login Intent Classifier
        
        Args:
            intents_file (str): Path to post-login intents JSON file
        """
        self.lemmatizer = WordNetLemmatizer()
        self.ignore_words = ['?', '!', '.', ',']
        
        # Load post-login specific intents
        with open(intents_file, 'r') as f:
            self.intents = json.load(f)
        
        # Initialize preprocessing attributes
        self.words = []
        self.classes = []
        self.documents = []
        
        # Label Encoder
        self.label_encoder = LabelEncoder()
        
    def preprocess_data(self):
        """
        Preprocess the post-login intent data
        """
        # Reset lists
        self.words = []
        self.classes = []
        self.documents = []
        
        # Process post-login specific intents
        for intent in self.intents['intents']:
            for pattern in intent['patterns']:
                # Modified: Use word_tokenize directly without language specification
                words = word_tokenize(pattern.lower())
                words = [self.lemmatizer.lemmatize(word) for word in words if word not in self.ignore_words]
                
                # Add to documents
                self.documents.append((words, intent['tag']))
                
                # Collect words
                self.words.extend(words)
                
                # Collect classes
                if intent['tag'] not in self.classes:
                    self.classes.append(intent['tag'])
        
        # Remove duplicates and sort
        self.words = sorted(list(set(self.words)))
        self.classes = sorted(list(set(self.classes)))
        
        # Encode labels
        self.label_encoder.fit(self.classes)
        
        # Save preprocessed data specifically for post-login
        with open('login_texts.pkl', 'wb') as f:
            pickle.dump(self.words, f)
        with open('login_labels.pkl', 'wb') as f:
            pickle.dump(self.classes, f)
        
        return self
    
    def create_training_data(self):
        """
        Create training data with bag of words approach for post-login intents
        """
        training_sentences = []
        training_labels = []
        
        for doc in self.documents:
            # Bag of words
            bag = [1 if w in doc[0] else 0 for w in self.words]
            
            # One-hot encode labels
            label = self.label_encoder.transform([doc[1]])[0] # type: ignore
            
            training_sentences.append(bag)
            training_labels.append(label)
        
        # Convert to numpy arrays
        X = np.array(training_sentences)
        y = keras.utils.to_categorical(training_labels)
        
        return X, y
    
    def build_model(self, input_shape):
        """
        Build an advanced Neural Network model for post-login intents
        """
        model = Sequential([
            Dense(256, input_shape=(input_shape,), activation='relu'),
            BatchNormalization(),
            Dropout(0.6),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dense(len(self.classes), activation='softmax')
        ])
        
        # Compile with advanced optimizer
        model.compile(
            loss='categorical_crossentropy', 
            optimizer=Adam(learning_rate=0.001),
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self, epochs=300, validation_split=0.2):
        """
        Train the post-login intent classification model
        """
        # Preprocess data
        self.preprocess_data()
        
        # Create training data
        X, y = self.create_training_data()
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, 
            test_size=validation_split, 
            random_state=42
        )
        
        # Build model
        model = self.build_model(X.shape[1])
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss', 
            patience=50,  # Increased patience to allow more epochs
            restore_best_weights=True,
            min_delta=0.001  # Minimum change to qualify as an improvement
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.2, 
            patience=10,  # Slightly increased patience
            min_lr=0.00001
        )
        
        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=16,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # Save model specifically for post-login
        model.save('login_intent_model.h5')
        
        return model, history
    
    def evaluate_model(self, model, X_test, y_test):
        """
        Evaluate the trained model
        """
        loss, accuracy = model.evaluate(X_test, y_test)
        print(f"Test Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")
        return loss, accuracy

def main():
    # Ensure data.json exists - create a sample if needed
    if not os.path.exists('data.json'):
        print("Creating sample data.json file...")
        sample_data = {
            "intents": [
                {
                    "tag": "appointment_booking",
                    "patterns": [
                        "I need to book an appointment",
                        "Schedule a visit with my doctor",
                        "Make an appointment",
                        "Book a slot with Dr. Smith"
                    ],
                    "responses": ["I'll help you book an appointment."]
                },
                {
                    "tag": "view_appointments",
                    "patterns": [
                        "Show me my appointments",
                        "When is my next appointment",
                        "List all my upcoming visits",
                        "View my schedule"
                    ],
                    "responses": ["Here are your upcoming appointments."]
                },
                {
                    "tag": "medication_reminder",
                    "patterns": [
                        "Remind me to take my medicine",
                        "Set a medication alert",
                        "I need a reminder for my pills",
                        "Create a med reminder"
                    ],
                    "responses": ["I'll set up a medication reminder for you."]
                },
                {
                    "tag": "health_info",
                    "patterns": [
                        "What is diabetes",
                        "Tell me about hypertension",
                        "Information about asthma",
                        "Facts about heart disease"
                    ],
                    "responses": ["Here's some information about that condition."]
                }
            ]
        }
        with open('data.json', 'w') as f:
            json.dump(sample_data, f, indent=4)
    
    # Initialize and train classifier for post-login intents
    print("Training post-login intent classifier...")
    classifier = PostLoginIntentClassifier()
    
    # Train model
    model, history = classifier.train_model()
    
    print("Post-login intent model trained successfully!")
    
    # Example prediction
    test_texts = [
        "I need to book an appointment",
        "Show me my upcoming visits",
        "What is diabetes",
        "Set a reminder for my medication"
    ]
    
    for text in test_texts:
        # Load the model and supporting files
        model = keras.models.load_model('login_intent_model.h5')
        words = pickle.load(open('login_texts.pkl', 'rb'))
        classes = pickle.load(open('login_labels.pkl', 'rb'))
        
        # Preprocess input
        lemmatizer = WordNetLemmatizer()
        sentence_words = word_tokenize(text.lower())
        sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words if word not in classifier.ignore_words]
        
        # Create bag of words
        bag = [1 if w in sentence_words else 0 for w in words]
        
        # Predict
        pred = model.predict(np.array([bag]))[0]
        intent_index = np.argmax(pred)
        intent = classes[intent_index]
        confidence = pred[intent_index]
        
        print(f"\nText: '{text}'")
        print(f"Predicted Intent: {intent}")
        print(f"Confidence: {confidence*100:.2f}%")

if __name__ == "__main__":
    main()