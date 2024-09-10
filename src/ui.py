import pickle
import numpy as np
import tkinter as tk
from tkinter import messagebox
from preprocessing import DataLoader
import keras

class FakeNewsApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fake News Detector")

        # Label for instruction
        self.label_input = tk.Label(root, text="Enter the text:")
        self.label_input.pack(pady=10)

        # Textbox for user input (multiline)
        self.text_input = tk.Text(root, height=10, width=50)
        self.text_input.pack(pady=10)

        # Button to classify the text
        self.button = tk.Button(root, text="Check", command=self.check_fake_news)
        self.button.pack(pady=10)

        # Label to display the result
        self.label_result = tk.Label(root, text="Result will be displayed here.", font=("Arial", 14))
        self.label_result.pack(pady=20)
        self.load_model()
    
    def load_model(self, model_path="../data/model.keras"):
        try:
            self.model = keras.models.load_model(model_path)
            self.saved_data = pickle.load(open("../data/data.pkl", "rb"))
            print(self.saved_data.keys())
        except Exception as e:
            messagebox.showerror("Error", f"Error loading the model: {e}")
            self.root.destroy()
        
        print("Model loaded successfully.")


    def check_fake_news(self):
        # Get the text input
        user_text = self.text_input.get("1.0", "end-1c")
        tokenized = DataLoader.tokenize_with_existing_tokenizer(user_text, self.saved_data["tokenizer"], self.saved_data["max_len"], self.saved_data["stopwords"])
        prediction = self.model.predict(np.array([tokenized]))
        if prediction < 0.5:
            self.label_result.config(text="Fake News", fg="red")
        else:
            self.label_result.config(text="Real News", fg="green")
        

# Create the Tkinter window
root = tk.Tk()

# Initialize and run the app
app = FakeNewsApp(root)
root.mainloop()