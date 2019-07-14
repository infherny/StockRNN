## StockRNN

Réseau récurrent pour prédire la tendance du cours d'une action.


Pour utiliser : 

Envirommenent : 

    `Conda create --name {name} python=3.7 anaconda` 
    `conda activate {name}` 
    `conda install theano` 
    `conda install tensorflow` 
    `conda install keras` `conda update --all`

Prédire la tendance d'un cours d'action : 

L'exemple suivant est avec l'action de Hexo.To

    hexo_rnn = StockRNN('Hexo', recurrence=120)
    hexo_rnn.load_data("hexo_train.csv", "hexo_test.csv")
    hexo_rnn.create_model(lstm_layers = 5)
    hexo_rnn.training(training_step=5)
    
