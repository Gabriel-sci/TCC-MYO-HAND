# TECNOLOGIA ASSISTIVA PARA AMPUTADOS: PRÓTESE MIOELÉTRICA CONTROLADA POR REDES NEURAIS ARTIFICIAIS
![image](https://github.com/user-attachments/assets/1b6a1aca-88c6-461f-bb5d-9a16e0383700)


# O Projeto:

O objetivo deste trabalho é prototipar um sistema embarcado para controle de uma prótese mioelétrica de mão, via CNN (Convolutional Neural Network), alimentada por sinais de tensão provenientes dos músculos do antebraço e braço, com alto grau de confiabilidade e que seja de baixo custo quando comparada a média do mercado, de forma a comprovar a possibilidade de democratizar o acesso a esta tecnologia com a população de países de baixa renda. Para a aquisição dos sinais de controle deve ser utilizado um sensor EMG AD8221, conectado a pele por meio de eletrodos de contato, que envia estes dados para um conversor A/D ADS1115, antes de serem enfim processados pelo microcomputador Raspberry Pi4, o qual deve processar um modelo tensorflow lite, treinado para identificar 3 classes de saída (descanso, flexão e extensão) e ativar os servomotores de acordo com o movimento correspondente, utilizando as GPIOs do Raspberry. São apresentados resultados de treinamento do modelo de deep learning, como curva ROC e matriz de confusão, comprovando sua eficiência, boa velocidade de inferência em novos dados, e tipos de movimentos possíveis de serem realizados com a prótese construída. Além disso, busca-se prever possíveis melhorias e ajustes para a continuidade e otimização do sistema.

# Instruções de uso:

O código "inference+servo_control.py" realiza a inferencia do modelo treinado e exportado para Tensorflow Lite: "model_cnn_emg.tflite", basta indicar o diretorio do modelo no código. 

O código realiza outras funções como a plotagem dos graficos de amplitude do sinal de envoltoria proveniente dos sensores EMG e ativação dos servomotores correspondentes a cada classificação da CNN. 

A placa de aquisição de dados pode ser impressa em PCB, seu arquivo schematic e design estão no zip "pcb-eagle.7z"

O modelo 3D utilizado na protese foi a Humanoid Robotic Hand de Grossrc, que tem seus arquivos .stl a disposição no Thingiverse: https://www.thingiverse.com/thing:2269115 
