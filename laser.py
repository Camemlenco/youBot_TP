import time
import math
import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

from hokuyo import HokuyoSensorSim 

def main():
    # 1. Configuração Inicial
    client = RemoteAPIClient()
    sim = client.getObject('sim')
    
    print("Conectado. Iniciando leitura do laser...")
    client.setStepping(True)
    sim.startSimulation()

    # 2. Instanciando o Sensor
    # Verifique na Scene Hierarchy onde o sensor está aninhado.
    try:
        # Se o sensor estiver dentro do youbot:
        laser_path = '/youBot/fastHokuyo'
        laser = HokuyoSensorSim(sim, laser_path)
    except ValueError as e:
        print(f"Erro ao inicializar: {e}")
        print("Dica: Verifique se o nome '/youBot/fastHokuyo' existe na sua cena.")
        sim.stopSimulation()
        return

    try:
        while True:
            time.sleep(1)  # Pequena pausa para estabilidade
            # 3. Leitura dos Dados
            # Retorna uma matriz N x 2: [[angulo, distancia], [angulo, distancia], ...]
            data = laser.getSensorData()
            
            # Se não houver dados válidos, pula o loop
            if len(data) == 0:
                client.step()
                continue

            # 4. Processamento: Encontrar o objeto mais próximo
            # Separa as colunas
            angles = data[:, 0]   # Primeira coluna: Ângulos (radianos)
            ranges = data[:, 1]   # Segunda coluna: Distâncias (metros)
            
            # Filtro básico: Ignorar leituras muito próximas (ruído do próprio robô) 
            # ou muito distantes (infinito/teto máximo do sensor)
            valid_mask = (ranges > 0.1) & (ranges < 4.0)
            
            if np.any(valid_mask):
                # Pega apenas os dados válidos
                valid_ranges = ranges[valid_mask]
                valid_angles = angles[valid_mask]
                
                # Encontra o índice da menor distância (o objeto mais perto)
                min_idx = np.argmin(valid_ranges)
                
                dist_objeto = valid_ranges[min_idx]
                angle_objeto = valid_angles[min_idx]
                
                # 5. Conversão Polar -> Cartesiana
                # X = Frente do robô, Y = Esquerda do robô
                obj_x = dist_objeto * math.cos(angle_objeto)
                obj_y = dist_objeto * math.sin(angle_objeto)
                
                # Formatação para printar
                graus = math.degrees(angle_objeto)
                
                print(f"--- OBJETO DETECTADO ---")
                print(f"Distância: {dist_objeto:.3f} m")
                print(f"Ângulo:    {graus:.1f} graus")
                print(f"Posição Relativa (X, Y): ({obj_x:.3f}, {obj_y:.3f})")
                
                # Lógica simples de quadrantes
                if obj_x > 0:
                    posicao = "Frente"
                else:
                    posicao = "Atrás"
                    
                if obj_y > 0.1:
                    posicao += "-Esquerda"
                elif obj_y < -0.1:
                    posicao += "-Direita"
                else:
                    posicao += "-Centro"
                    
                print(f"Setor: {posicao}")
                print("-" * 30)
            
            else:
                print("Nenhum objeto próximo detectado.")

            # Avança o tempo
            client.step()

    except KeyboardInterrupt:
        print("Parando...")
    finally:
        sim.stopSimulation()

if __name__ == "__main__":
    main()