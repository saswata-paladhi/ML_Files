import numpy as np
import pandas as pd
import yfinance as yf
from datetime import date
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import streamlit as st
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim= hidden_dim
        self.num_layers= num_layers
        self.lstm= nn.LSTM(input_dim, hidden_dim, num_layers, batch_first= True)
        self.fc= nn.Linear(hidden_dim, output_dim)          #fully connected layer after lstm
    def forward(self, x):
        h0= torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().cuda()               #.requires_grad is for computing gradients
        c0= torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().cuda()
        out, (hn,cn)= self.lstm(x, (h0.detach(), c0.detach()))
        out= self.fc(out[:,-1,:])           #This represents the final state of the lstm as the output.
        return out
st.set_page_config(
	page_title="Stock Price Prediction",
	layout="wide"
)


def get_ticker(name):
	company = yf.Ticker(name) # google
	return company

st.title("Stock Price Plotting")
st.header("Plotting Adjusted Close Price for requested companies:")
selected_company = st.text_input('Ticker Code')
feature = st.text_input('Type of price')

five_yrs_ago = date.today() - relativedelta(years=5)

if st.button('Display'):
    df_train= yf.download(selected_company, start=five_yrs_ago.strftime("%Y-%m-%d"), end= date.today().strftime("%Y-%m-%d"))
    df= df_train.loc[:, feature]
    df= df.values.reshape(-1,1)
    scaler= MinMaxScaler(feature_range=(-1,1))
    scaled_df= scaler.fit_transform(df)
    x_train= []
    y_train= []
    time_step= 100
    for i in range(time_step, len(scaled_df)):
        x_train.append(scaled_df[i-time_step:i,0])
        y_train.append(scaled_df[i,0])
    x_train, y_train= np.array(x_train), np.array(y_train)
    x_train= np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1))
    device= torch.device('cuda')
    x_train= torch.from_numpy(x_train).type(torch.Tensor)
    y_train= torch.from_numpy(y_train).type(torch.Tensor)
    y_train= np.reshape(y_train, (y_train.shape[0], 1))
    input_dim=1
    hidden_dim_lstm= 16
    num_layers_lstm= 2         #number of stacks of lstm
    output_dim= 1
    epochs_lstm= 1000

    model_lstm= LSTM(input_dim= input_dim, hidden_dim= hidden_dim_lstm, output_dim= output_dim, num_layers= num_layers_lstm)
    criterion= torch.nn.MSELoss(reduction= 'mean')
    optimizer= torch.optim.Adam(model_lstm.parameters(), lr= 0.01)
    model_lstm.cuda()
    for e in range(epochs_lstm):
        y_train_pred= model_lstm(x_train.cuda())
        loss= criterion(y_train_pred, y_train.cuda())
        optimizer.zero_grad()
        loss.backward()         #Backpropogation
        optimizer.step()        #Gradient descent step
    model_lstm.train()

    company = get_ticker(selected_company)
    st.write("""
	    ### {}
	    """.format(company.info['shortName']))
    st.write(company.info['longBusinessSummary'])

    y_train_pred_lstm= model_lstm(x_train.cuda())
    st.write(""" #### Past 5 years Data:""")
    st.line_chart(data = df_train[feature], width = 800, height = 300, use_container_width = False)
    pred_lstm= scaler.inverse_transform(y_train_pred_lstm.cpu().detach().numpy())
    price= '{:.2f}'.format(pred_lstm[-1,0])
    st.write(f'\nTomorrow\'s {feature} will be: \n ### {price}')