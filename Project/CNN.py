# -*- coding: utf-8 -*-
"""
   Created on Sat Nov 23 22:52:29 2019

@author: Dhruv
"""
import numpy as np
import random as rn

NO_OF_ACTIONS=6


NO_OF_FILTER_CONV1=16
NO_OF_FILTER_CONV2=32
NO_OF_INPUT_HEIGHT=84
NO_OF_INPUT_WIDTH=84
INPUT_CHANNELS=4
STRIDE_CONV1=4
STRIDE_CONV2=2
LEARNING_RATE=0.00025
RHO=0.95
EPSILON=0.01
MINI_BATCH_SIZE=32
FILTER_DIM_CONV1=(8,8)
FILTER_DIM_CONV2=(4,4)
NO_OF_NEURONS_FCC=256
NO_INPUTS_FCC=2592
flag=0    # totrack whether weights areinitializing for first time
NO_OF_ITER=100

filter_conv1=filter_conv2=weight_fcc=weight_output=[]


def relu_derivative(x):
    #
    #print('before relu ', x)
    x[x<=0] = 0
    x[x>0] = 1
    #print('  after relu  ',x)
    return x;

class convolutionLayer:
    def __init__(self,no_of_filters,filter_rows,filter_column, stride):
        self.no_of_filters=no_of_filters
        self.no_rows=filter_rows
        self.no_columns=filter_column
        self.stride=stride
        #filter = np.random.randn(INPUT_HIEGHT, filter_rows, filter_column) / (filter_rows*filter_column)  # Xavier Inittialization
        
        print()
    def convolve(self,input, filter):
        #print(input.shape)
        
        input_depth,input_rows,input_colums=input.shape
        output_conv=np.zeros(((self.no_of_filters),(((input_rows - self.no_rows)//self.stride)+1),(((input_colums - self.no_columns )//self.stride)+1)))
        l=0
        filters=filter
        self.filter=filter
        while l<self.no_of_filters:
            #self.filter = ((np.random.randn(input_depth, self.no_rows, self.no_columns) / (self.no_rows*self.no_columns))+2).round()  # Xavier Inittialization
            #print('Filter')
            #print(self.filter)
            #filters.append(self.filter)
    # =============================================================================
    #         input[0,0,0]=0
    #         input[0,0,1]=1
    #         input[0,0,2]=2
    #         input[0,1,0]=3
    #         input[0,1,1]=4
    #         input[0,1,2]=5
    #         input[0,2,0]=6
    #         input[0,2,1]=7
    #         input[0,2,2]=8
    #         
    #         input[1,0,0]=0
    #         input[1,0,1]=1
    #         input[1,0,2]=2
    #         input[1,1,0]=3
    #         input[1,1,1]=4
    #         input[1,1,2]=5
    #         input[1,2,0]=6
    #         input[1,2,1]=7
    #         input[1,2,2]=8
    # =============================================================================
            
            #print(input)
            i=0
            j=0
            k=m=n=0
            # i= depth, j= rows,k=columns
            #while l<self.no_of_filters:
            while j+self.no_rows<=input_rows:
                while i<input_depth:
#                    print("first",input[i])
#                    print("seco",input[i,j:j+3,:])
#                    print("thir",(input[i,j:j+3,k:k+3]))
#                    
#                    print(i,' ',j,' ',k,' ',l,' ',m,' ',n,' ')
                    output_conv[l,m,n]+=np.sum(input[i,j:j+self.no_rows,k:k+self.no_columns]*self.filter[l,i,:,:])
                    #print(i,' ',j,' ',k,' ',l,' ',m,' ',n,' ')
                    print()
                    #print(output_conv[m][n])
                    i=i+1
                print() 
                #print(output_conv)
                
                k=k+self.stride
                i=0
                n=n+1
                if k+self.no_columns>input_colums:
                    
                    k=0
                    j=j+self.stride
                    m=m+1
                    n=0
                    
                    
            l=l+1
        print()
#        print('Final output matrix before relu')
#        print(output_conv)
        output_conv= np.maximum(0,output_conv)
#        print('Final output matrix after relu')
#        print(output_conv)
        return output_conv, filters
    
    def backPropagate(self,filters,delta_next_layer,stride,input):
        
       # print('Conv2 output shape ', delta_next_layer.shape)
        i,j,k= delta_next_layer.shape   # Layer output dimensions
        l,m,n=filters[0].shape # dimension of a single filter
        
        delta_filters=np.zeros(filters.shape)
        delta= np.zeros(input.shape)
        index_row=index_column=0
        for n_f in range(i):
            for input_height in range(j):
                for input_width in range(k):
                    delta_filters[n_f]+=input[:,index_row:index_row+m,index_column:index_column+n]*delta_next_layer[n_f,input_height,input_width]
                    delta[:,index_row:index_row+m,index_column:index_column+n]+=filters[n_f]*delta_next_layer[n_f,input_height,input_width]
                    index_column+=stride
                index_row+=stride
                index_column=0
                
        
        return delta,delta_filters
                    
        
        
        
        
class fullyConnectedLayer:
    
    def __init__(self,no_of_neurons,input,weight):
        self.no_of_neurons=no_of_neurons
        self.input_fcc=input
        #self.weight=np.random.randn(len(self.input_fcc),no_of_neurons)
        self.weight=weight
        #print('FCC weight matrix ',self.weight)
        global weight_fcc
        weight_fcc=self.weight
        
    
    def forward(self):
        self.output= np.dot(self.input_fcc,self.weight )
        #print('line 103 output_FCC ',self.output)
        return np.maximum(0,self.output)
    
    def backpropagate(self,delta_output,weight_output):
        
        delta_fcc=delta_output.dot(weight_output.T) *relu_derivative(np.maximum(0,self.output))
        
        delta_fcc_weight=LEARNING_RATE * self.input_fcc.T.dot(delta_fcc)
        return delta_fcc, delta_fcc_weight
        

class outputLayer:
    
    def __init__(self,input,weight):
        self.no_of_actions=NO_OF_ACTIONS
        self.input_output_layer=input
        #print('input to output layer type ',type(self.input_output_layer))
        #print('input to output layershape  ',self.input_output_layer.shape,' length ',len(self.input_output_layer))
       # self.weight_output=np.random.randn(self.input_output_layer.shape[0],self.no_of_actions)  # 256x6
        #print('OL weight matrix ',self.weight_output)
        self.weight=weight
        
    
    def forward(self):
        
        #print('inside forward ',self.weight_output)
        
        global weight_output
        weight_output=self.weight_output
        return np.dot(self.input_output_layer,self.weight_output )
    
    def backpropagate(self,actual_output,pred_output):
        
        delta_output=actual_output-pred_output
        
        delta_output_weight=LEARNING_RATE * self.input_output_layer.T.dot(delta_output)
        return delta_output,delta_output_weight
        
        

def calculateError(pred_output,actual_output):
    return (np.sum(np.power((pred_output - actual_output), 2)))
    
def train(x,y,parameters):   # x, y = input and expected output of size 32
    
    global FILTER_DIM_CONV1,FILTER_DIM_CONV2,NO_OF_FILTER_CONV1, NO_OF_FILTER_CONV2, NO_OF_NEURONS_FCC
    filter_conv1,filter_conv2,weight_fcc,weight_output=parameters
    
   ############### Forward Phase  #############
    ########## 1st hidden layer  ############
    
    conv_layer1=convolutionLayer(NO_OF_FILTER_CONV1,FILTER_DIM_CONV1,STRIDE_CONV1)
    output_conv1=conv_layer1.convolve(x,filter_conv1)  # TO DO: change the input to the image
    
    ########### 2nd hidden layer ###########
    
    conv_layer2=convolutionLayer(NO_OF_FILTER_CONV2,FILTER_DIM_CONV2,STRIDE_CONV2)
    output_conv2=conv_layer2.convolve(output_conv1,filter_conv2)
    output_conv2_shape=output_conv2.shape
    input_fcc_hidden_layer=output_conv2.flatten()
    ######### FCC  ################
#    print('input to fcc layer ',input_fcc_hidden_layer )
#    print(type(input_fcc_hidden_layer),' ',input_fcc_hidden_layer.shape)
    
    fcc=fullyConnectedLayer(NO_OF_NEURONS_FCC,input_fcc_hidden_layer,weight_fcc)
    output_fcc=fcc.forward()
    #print("output FCC ",output_fcc)
    
    ###########  OUPUT LAYER ############
    ol=outputLayer(output_fcc,weight_fcc)
    final_output=ol.forward()
    #print("Final output  ",final_output)
    #print('output layer weights ',weight_output)
    
    error=calculateError(y,final_output)
    
    ############### Backpropagation ###################
    
    delta_output,delta_output_weight=ol.backpropagate(y,final_output)  # Output Layer
    delta_fcc,delta_fcc_weight=fcc.backpropagate(delta_output,weight_output)  # FCC Layer
    delta_conv2,delta_filter_conv2=conv_layer2.backpropagate(filter_conv2,delta_fcc.reshape(output_conv2_shape),STRIDE_CONV2,output_conv1)  # Conv2 Layer
    delta_conv1,delta_filter_conv1=conv_layer1.backpropagate(filter_conv2,delta_fcc,STRIDE_CONV2,x)  # Conv1 Layer
    
    grads=[delta_filter_conv1,delta_filter_conv2,delta_fcc_weight,delta_output_weight]
    return grads,error

def RMSProp(mini_batch,parameters):
    global LEARNING_RATE,RHO,EPSILON,error
    
    filter_conv1,filter_conv2,weight_fcc,weight_output=parameters
    
    
    error_sum = 0
    
    
    # initialize gradients
    df1 =s1= np.zeros(filter_conv1.shape)
    df2 =s2= np.zeros(filter_conv2.shape)
    dw_fcc=s3 = np.zeros(weight_fcc.shape)
    dw_ouput=s4 = np.zeros(weight_output.shape)
    for j in range(len(mini_batch)):
        x=mini_batch[j][0]
        y=mini_batch[j][1]
        print('line 270 ',x,' ',y)
    ## To do
        for i in range(len(x)):
            print('line 270 ',x[i],' ',y[i])
            grads,error=train(x[i],y[i])
            df1+=grads[0]
            df2+=grads[1]
            dw_fcc+=grads[2]
            dw_ouput+=grads[3]
            error_sum+=error
        s1 = RHO*s1 + (1-RHO)*((df1)**2 )# RMSProp update
        filter_conv1-=LEARNING_RATE*(df1/np.sqrt(s1)+EPSILON)
        s2 = RHO*s2 + (1-RHO)*((df2)**2 )# RMSProp update
        filter_conv2-=LEARNING_RATE*(df2/np.sqrt(s2)+EPSILON)
        s3 = RHO*s3 + (1-RHO)*((dw_fcc)**2 )# RMSProp update
        weight_fcc-=LEARNING_RATE*(dw_fcc/np.sqrt(s3)+EPSILON)
        s4 = RHO*s4 + (1-RHO)*((dw_ouput)**2 )# RMSProp update
        weight_output-=LEARNING_RATE*(dw_ouput/np.sqrt(s4)+EPSILON)


    return filter_conv1,filter_conv2,weight_fcc,weight_output,error
            
    
    
    
         
        
        

def get_minibatch(replay_buffer, y):
    global MINI_BATCH_SIZE
    minibatches = []

    X, y = rn.shuffle(replay_buffer, y)

    for i in range(0, replay_buffer.shape[0]):
        X_mini = X[i:i + MINI_BATCH_SIZE]
        y_mini = y[i:i + MINI_BATCH_SIZE]

        minibatches.append((X_mini, y_mini))

    return minibatches

def initialiseNetwork(replay_buffer,y):
# =============================================================================
#     global NO_OF_ACTIONS,NO_OF_FILTER_CONV1,NO_OF_FILTER_CONV2,INPUT_HEIGHT,INPUT_WIDTH, INPUT_CHANNELS,STRIDE_CONV1, STRIDE_CONV2, MINI_BATCH_SIZE, FILTER_DIM_CONV1
    global FILTER_DIM_CONV2,NO_OF_NEURONS_FCC, NO_INPUTS_FCC,filter_conv1,filter_conv2,weight_fcc,weight_output
#     
#     filter_conv1= ((np.random.randn(NO_OF_FILTER_CONV1,INPUT_CHANNELS,FILTER_DIM_CONV1.shape[0],FILTER_DIM_CONV1.shape[1]) / (FILTER_DIM_CONV1.shape[0]*FILTER_DIM_CONV1.shape[1]))+2).round()  # Xavier Inittialization
#     filter_conv2=((np.random.randn(NO_OF_FILTER_CONV2,INPUT_CHANNELS, FILTER_DIM_CONV2.shape[0], FILTER_DIM_CONV2.shape[1]) / (FILTER_DIM_CONV2.shape[0]*FILTER_DIM_CONV2.shape[1]))+2).round() 
#     weight_fcc= np.random.randn(NO_INPUTS_FCC,NO_OF_NEURONS_FCC)
#     weight_output=np.random.randn(NO_OF_NEURONS_FCC,NO_OF_ACTIONS)
#     parameters=[filter_conv1,filter_conv2,weight_fcc,weight_output]
# =============================================================================
    
    parameters=getWeights()
    
    mini_batch=get_minibatch(replay_buffer,y)     # TO DO
    global NO_OF_ITER
    error=0
    
    for iter in range(1,NO_OF_ITER+1):
      
       parameters= RMSProp(mini_batch,parameters)
       filter_conv1,filter_conv2,weight_fcc,weight_output,err=parameters
       error+=err
    return (error/NO_OF_ITER)
       
    
   

def getWeights():
    global flag
    global NO_OF_ACTIONS,NO_OF_FILTER_CONV1,NO_OF_FILTER_CONV2,INPUT_HEIGHT,INPUT_WIDTH, INPUT_CHANNELS,STRIDE_CONV1, STRIDE_CONV2, MINI_BATCH_SIZE, FILTER_DIM_CONV1
    global FILTER_DIM_CONV2,NO_OF_NEURONS_FCC, NO_INPUTS_FCC,filter_conv1,filter_conv2,weight_fcc,weight_output
    if flag ==0:
        filter_conv1= ((np.random.randn(NO_OF_FILTER_CONV1,INPUT_CHANNELS,FILTER_DIM_CONV1[0],FILTER_DIM_CONV1[1]) / (FILTER_DIM_CONV1[0]*FILTER_DIM_CONV1[1]))+2).round()  # Xavier Inittialization
        filter_conv2=((np.random.randn(NO_OF_FILTER_CONV2,INPUT_CHANNELS, FILTER_DIM_CONV2[0], FILTER_DIM_CONV2[1]) / (FILTER_DIM_CONV2[0]*FILTER_DIM_CONV2[1]))+2).round() 
        weight_fcc= np.random.randn(NO_INPUTS_FCC,NO_OF_NEURONS_FCC)
        weight_output=np.random.randn(NO_OF_NEURONS_FCC,NO_OF_ACTIONS)
    return filter_conv1,filter_conv2,weight_fcc,weight_output
        
def predict(input):
    global flag
    if flag ==0:
        filter_conv1,filter_conv2,weight_fcc,weight_output=getWeights()
         ########## 1st hidden layer  ############
    
    conv_layer1=convolutionLayer(NO_OF_FILTER_CONV1,FILTER_DIM_CONV1,STRIDE_CONV1)
    output_conv1=conv_layer1.convolve(input,filter_conv1)  # TO DO: change the input to the image
    
    ########### 2nd hidden layer ###########
    
    conv_layer2=convolutionLayer(NO_OF_FILTER_CONV2,FILTER_DIM_CONV2,STRIDE_CONV2)
    output_conv2=conv_layer2.convolve(output_conv1,filter_conv2)
    output_conv2_shape=output_conv2.shape
    input_fcc_hidden_layer=output_conv2.flatten()
    ######### FCC  ################
#    print('input to fcc layer ',input_fcc_hidden_layer )
#    print(type(input_fcc_hidden_layer),' ',input_fcc_hidden_layer.shape)
    
    fcc=fullyConnectedLayer(NO_OF_NEURONS_FCC,input_fcc_hidden_layer,weight_fcc)
    output_fcc=fcc.forward()
    #print("output FCC ",output_fcc)
    
    ###########  OUPUT LAYER ############
    ol=outputLayer(output_fcc,weight_fcc)
    final_output=ol.forward()
    #print("Final output  ",final_output)
    return final_output
    #print('output layer weights ',weight_output)
    
def predict_target(input,parameters):
    
    print("line 382 ",parameters)
    
    filter_conv1,filter_conv2,weight_fcc,weight_output=parameters
         ########## 1st hidden layer  ############
    
    conv_layer1=convolutionLayer(NO_OF_FILTER_CONV1,FILTER_DIM_CONV1[0],FILTER_DIM_CONV1[1],STRIDE_CONV1)
    output_conv1=conv_layer1.convolve(input,filter_conv1)  # TO DO: change the input to the image
    
    ########### 2nd hidden layer ###########
    
    conv_layer2=convolutionLayer(NO_OF_FILTER_CONV2,FILTER_DIM_CONV2[0],FILTER_DIM_CONV2[1],STRIDE_CONV2)
    output_conv2=conv_layer2.convolve(output_conv1,filter_conv2)
    output_conv2_shape=output_conv2.shape
    input_fcc_hidden_layer=output_conv2.flatten()
    ######### FCC  ################
    #print('input to fcc layer ',input_fcc_hidden_layer )
    #print(type(input_fcc_hidden_layer),' ',input_fcc_hidden_layer.shape)
    
    fcc=fullyConnectedLayer(NO_OF_NEURONS_FCC,input_fcc_hidden_layer,weight_fcc)
    output_fcc=fcc.forward()
    #print("output FCC ",output_fcc)
    
    ###########  OUPUT LAYER ############
    ol=outputLayer(output_fcc,weight_fcc)
    final_output=ol.forward()
    #print("Final output  ",final_output)
    return final_output
    
        
    
    
         
                
                
                
                
