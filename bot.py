import requests
import os
import json
import time
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np

# Replace with your bot's token and chat ID
TOKEN = '6778057365:AAFjgnrCsR6H0oma8PjFiul_zEURalD6If8'
CHAT_ID = '761123656'

# Function to send message via Telegram bot
def send_telegram_message(message):
    url = f'https://api.telegram.org/bot{TOKEN}/sendMessage'
    data = {'chat_id': CHAT_ID, 'text': message}
    response = requests.post(url, data=data)
    return response.json()

# Function to read log file and extract relevant info
def read_log_file():
    with open('./logs/frcnn/train.log', 'r') as file:
        lines = file.readlines()

    # Find the latest epoch and loss information in the log file
    latest_epoch = None
    latest_loss = None

    for line in reversed(lines):
        if 'Epoch' in line and 'Loss' in line:
            # Extract epoch and loss information
            parts = line.split('Epoch: [')[1].split('] [')
            epoch = int(parts[0])
            loss = float(parts[1].split('Loss: ')[1])

            # Store the latest epoch and loss
            if latest_epoch is None:
                latest_epoch = epoch
                latest_loss = loss
                break

    return latest_epoch, latest_loss

# Function to draw and save charts
def draw_charts(epochs, losses):
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=epochs, y=losses, marker='o', color='blue', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    
    # Save the chart as an image file
    plt.savefig('training_loss_chart.png')
    plt.close()

# Main loop for monitoring
def monitor_training():
    epochs, losses = [], []
    while True:
        epoch, loss = read_log_file()
        
        if epoch and epoch >= 199:
            draw_charts(epochs, losses)
            # send message
            send_telegram_message("Training Ended.\nHere are the preliminary results.")
            # Send the chart image to Telegram
            url = f'https://api.telegram.org/bot{TOKEN}/sendPhoto'
            files = {'photo': open('training_loss_chart.png', 'rb')}
            data = {'chat_id': CHAT_ID}
            response = requests.post(url, files=files, data=data)
            print(response.json())  # Print the response (optional)
            
            # You can also remove the image file after sending if needed
            os.remove('training_loss_chart.png')
            
            # Break the loop after sending the charts
            break

        

        # Send updates to Telegram
        if epoch and loss:
            message = f"Epoch: {epoch}, Loss: {loss}"
            send_telegram_message(message)
            epochs.append(epoch)
            losses.append(loss)

        # Sleep for a certain interval before checking the log again
        time.sleep(300)  # Sleep for 5 minutes

if __name__ == "__main__":
    monitor_training()
