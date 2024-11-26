import pigpio
import time
import board
import busio
import numpy as np
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn
import tflite_runtime.interpreter as tflite
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import MultipleLocator

# Configuração do I2C e do ADS1115
i2c = busio.I2C(board.SCL, board.SDA)
ads = ADS.ADS1115(i2c)
ads.gain = 1  # Configuração de ganho para leituras de até ~6V
ads.data_rate = 860  # Define a taxa de amostragem diretamente no ADS1115

# Carrega o modelo TensorFlow Lite
interpreter = tflite.Interpreter(model_path='/home/gabriel/modelo_cnn_emg (4).tflite')
interpreter.allocate_tensors()
input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

# Número de amostras esperado pelo modelo (ajustado para 50)
amostras_por_janela = 50

# Configuração para o gráfico
fig, ax = plt.subplots()
ax.set_ylim(0, 5)  # Limite de 0 a 6V com margem
ax.set_xlabel('Tempo (s)')
ax.set_ylabel('Tensão (V)')
ax.set_title('Leituras dos Canais EMG')
ax.yaxis.set_major_locator(MultipleLocator(0.5))
ax.xaxis.set_major_locator(MultipleLocator(5))

# Listas para armazenar os dados de cada canal
tempos = []
leituras_canal0, leituras_canal2, leituras_canal3 = [], [], []
tempo_inicial = time.time()

# Função para coleta dos dados com normalização
def coletar_dados_emg():
    canal0 = AnalogIn(ads, ADS.P0)
    canal2 = AnalogIn(ads, ADS.P2)
    canal3 = AnalogIn(ads, ADS.P3)
    return [canal0.voltage * 2, canal2.voltage * 2, canal3.voltage * 2]

# Configuração dos pinos GPIO para os servos com pigpio
pinos_servos = [17, 18, 27, 22, 23]
pinos_selecionados = [22, 23, 18]  # Servos específicos a serem ativados com GPIO 24 HIGH
gpio_controle = 24  # Pino de controle

pi = pigpio.pi()
if not pi.connected:
    raise RuntimeError("Falha ao conectar ao daemon pigpio")

def configurar_gpio(pinos, gpio_controle):
    for pino in pinos:
        pi.set_mode(pino, pigpio.OUTPUT)
    pi.set_mode(gpio_controle, pigpio.INPUT)

configurar_gpio(pinos_servos, gpio_controle)

# Função para rotacionar os servos
def rotacionar_servos(angulo, pinos):
    pwm = 500 + (angulo / 180) * 2000  # Conversão de ângulo para largura do pulso (500-2500 µs)
    for pino in pinos:
        pi.set_servo_pulsewidth(pino, pwm)

# Estado inicial e controle por repetição
estado_servo = None
contador_predicao = 0
ultima_classe_predita = None

# Atualização do gráfico e controle dos servos
# Atualização do gráfico e controle dos servos
def atualizar(frame):
    global tempos, leituras_canal0, leituras_canal2, leituras_canal3
    global estado_servo, contador_predicao, ultima_classe_predita

    amostras = []
    for _ in range(amostras_por_janela):
        leituras = coletar_dados_emg()
        amostras.append(leituras)
        leituras_canal0.append(leituras[0])
        leituras_canal2.append(leituras[1])
        leituras_canal3.append(leituras[2])
        tempos.append(time.time() - tempo_inicial)
        time.sleep(1 / ads.data_rate)

    # Plotagem das leituras dos canais
    ax.clear()
    ax.plot(tempos, leituras_canal0, label='Canal 0', color='blue')
    ax.plot(tempos, leituras_canal2, label='Canal 2', color='green')
    ax.plot(tempos, leituras_canal3, label='Canal 3', color='red')
    ax.legend()
    ax.set_ylim(0, 6.5)
    ax.set_xlim(max(0, tempos[-1] - 5), tempos[-1] + 1)

    # Realizar a predição
    dados_entrada = np.array([amostras], dtype=np.float32)
    interpreter.set_tensor(input_index, dados_entrada)
    interpreter.invoke()
    predicao = interpreter.get_tensor(output_index)
    classe_predita = np.argmax(predicao)

    # Verificar repetição da mesma classe
    if classe_predita == ultima_classe_predita:
        contador_predicao += 1
    else:
        contador_predicao = 1  # Resetar contador se a classe mudar
        ultima_classe_predita = classe_predita

    # Verificar estado do GPIO 24 após cada predição
    estado_gpio = pi.read(gpio_controle)

    # Atuar nos servos apenas após 4 predições consecutivas
    if contador_predicao > 2:
        if estado_gpio == 1:
            # GPIO HIGH: Ativar apenas os servos das portas 22, 23 e 18
            pinos_ativos = pinos_selecionados
            if classe_predita == 1 and estado_servo != 40:
                rotacionar_servos(10, [22])
                rotacionar_servos(40, [18])
                rotacionar_servos(70, [23])
                estado_servo = 40
                print("Servos girados para 40 graus (GPIO HIGH, Classe 1)")
            elif classe_predita == 2 and estado_servo != 180:
                rotacionar_servos(180, pinos_ativos)
                estado_servo = 180
                print("Servos retornados para 180 graus (GPIO HIGH, Classe 2)")
            elif classe_predita == 0:
                print("Classe 0 detectada, servos mantidos na posição atual (GPIO HIGH).")
        else:
            # GPIO LOW: Ativar todos os servos
            pinos_ativos = pinos_servos
            if classe_predita == 1 and estado_servo != 0:
                rotacionar_servos(0, pinos_ativos)
                estado_servo = 0
                print("Servos girados para 0 graus (GPIO LOW, Classe 1)")
            elif classe_predita == 2 and estado_servo != 180:
                rotacionar_servos(180, pinos_ativos)
                estado_servo = 180
                print("Servos retornados para 180 graus (GPIO LOW, Classe 2)")
            elif classe_predita == 0:
                print("Classe 0 detectada, servos mantidos na posição atual (GPIO LOW).")

                
    # Exibir a classe predita e confiança no terminal
    print(f"Classe Predita: {classe_predita}, Confiança: {predicao[0][classe_predita]:.2f}")

# Cleanup ao finalizar
def finalizar():
    for pino in pinos_servos:
        pi.set_servo_pulsewidth(pino, 0)  # Desativa os servos
    pi.stop()
    print("GPIO limpo e finalizado.")

# Animação do gráfico
try:
    ani = FuncAnimation(fig, atualizar, interval=1000/860)
    plt.show()
except KeyboardInterrupt:
    finalizar()
