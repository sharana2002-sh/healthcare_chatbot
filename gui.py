# -*- coding: utf-8 -*-
# Import necessary libraries
from tkinter import *
import time
import tkinter.messagebox
import nltk
import random
import numpy as np
import json
import pickle
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load chatbot model and data
with open('intents.json') as json_file:
    intents = json.load(json_file)

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbotmodel.h5')

# Define chatbot logic
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    if intents_list:
        tag = intents_list[0]['intent']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                return random.choice(i['responses'])
    return "I'm sorry, I didn't understand that."

def chat(user_input):
    intents_list = predict_class(user_input)
    return get_response(intents_list, intents)

# GUI Code
class ChatInterface(Frame):

    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master
        self.tl_bg = "#EEEEEE"
        self.tl_bg2 = "#EEEEEE"
        self.tl_fg = "#000000"
        self.font = "Verdana 10"

        # Menu bar
        menu = Menu(self.master)
        self.master.config(menu=menu, bd=5)
        file = Menu(menu, tearoff=0)
        menu.add_cascade(label="File", menu=file)
        file.add_command(label="Clear Chat", command=self.clear_chat)
        file.add_command(label="Exit", command=self.chatexit)

        help_option = Menu(menu, tearoff=0)
        menu.add_cascade(label="Help", menu=help_option)
        help_option.add_command(label="About", command=self.msg)

        # Chat area
        self.text_frame = Frame(self.master, bd=6)
        self.text_frame.pack(expand=True, fill=BOTH)
        self.text_box_scrollbar = Scrollbar(self.text_frame, bd=0)
        self.text_box_scrollbar.pack(fill=Y, side=RIGHT)
        self.text_box = Text(self.text_frame, yscrollcommand=self.text_box_scrollbar.set, state=DISABLED,
                             bd=1, padx=6, pady=6, wrap=WORD, bg="#FFFFFF", font="Verdana 10", relief=GROOVE)
        self.text_box.pack(expand=True, fill=BOTH)
        self.text_box_scrollbar.config(command=self.text_box.yview)

        # Entry box and send button
        self.entry_frame = Frame(self.master, bd=1)
        self.entry_frame.pack(side=LEFT, fill=BOTH, expand=True)
        self.entry_field = Entry(self.entry_frame, bd=1, justify=LEFT)
        self.entry_field.pack(fill=X, padx=6, pady=6, ipady=3)
        self.send_button = Button(self.entry_frame, text="Send", width=5, relief=GROOVE, bg='white',
                                  command=lambda: self.send_message_insert(None))
        self.send_button.pack(side=RIGHT, padx=6, pady=6)
        self.master.bind("<Return>", self.send_message_insert)

        # Display welcome message and info
        self.msg()  # Show the About message as a pop-up
        self.display_initial_message()  # Display the welcome message in the chatbox

    def clear_chat(self):
        self.text_box.config(state=NORMAL)
        self.text_box.delete(1.0, END)
        self.text_box.config(state=DISABLED)

    def chatexit(self):
        exit()

    def msg(self):
        tkinter.messagebox.showinfo("MedBot", "MedBot is a chatbot for answering health-related queries.")

    def display_initial_message(self):
        self.text_box.config(state=NORMAL)
        self.text_box.insert(END, "MedBot: Hello! I am MedBot, your medical assistant. How can I help you today?\n")
        self.text_box.config(state=DISABLED)

    def send_message_insert(self, message):
        user_input = self.entry_field.get()
        if user_input.strip() == "":
            return
        self.text_box.config(state=NORMAL)
        self.text_box.insert(END, "You: " + user_input + "\n")
        self.text_box.config(state=DISABLED)
        self.text_box.see(END)
        self.entry_field.delete(0, END)

        bot_response = chat(user_input)
        self.text_box.config(state=NORMAL)
        self.text_box.insert(END, "MedBot: " + bot_response + "\n")
        self.text_box.config(state=DISABLED)
        self.text_box.see(END)

# Run the application
if __name__ == "__main__":
    root = Tk()
    root.title("MedBot")
    root.geometry("500x500")
    app = ChatInterface(master=root)
    app.pack(expand=True, fill=BOTH)
    root.mainloop()
