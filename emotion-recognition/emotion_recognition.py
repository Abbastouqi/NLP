import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class EmotionRecognition:
    def __init__(self):
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.model = self.build_model()
        
    def build_model(self):
        model = Sequential([
            Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)),
            Conv2D(64, kernel_size=(3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            Conv2D(128, kernel_size=(3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(128, kernel_size=(3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            Flatten(),
            Dense(1024, activation='relu'),
            Dropout(0.5),
            Dense(7, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self, train_dir, validation_dir, epochs=50):
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        validation_datagen = ImageDataGenerator(rescale=1./255)
        
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(48, 48),
            batch_size=64,
            color_mode='grayscale',
            class_mode='categorical'
        )
        
        validation_generator = validation_datagen.flow_from_directory(
            validation_dir,
            target_size=(48, 48),
            batch_size=64,
            color_mode='grayscale',
            class_mode='categorical'
        )
        
        history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator
        )
        
        return history
    
    def detect_emotion(self, frame):
        # Load the pre-trained face cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            # Extract face ROI
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            roi_gray = roi_gray.astype('float')/255.0
            roi_gray = np.expand_dims(roi_gray, axis=0)
            roi_gray = np.expand_dims(roi_gray, axis=-1)
            
            # Make prediction
            prediction = self.model.predict(roi_gray)
            emotion_label = self.emotions[np.argmax(prediction)]
            
            # Draw rectangle and label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, emotion_label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        return frame
    
    def start_webcam(self):
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = self.detect_emotion(frame)
            cv2.imshow('Emotion Recognition', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

def main():
    # Initialize emotion recognition system
    emotion_detector = EmotionRecognition()
    
    # If you have training data, you can train the model
    # emotion_detector.train_model('path/to/train/data', 'path/to/validation/data')
    
    # Start real-time emotion detection
    emotion_detector.start_webcam()

if __name__ == "__main__":
    main()