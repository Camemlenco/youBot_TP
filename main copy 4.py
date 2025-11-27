import time
import math
import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# Importa a classe do laser
from hokuyo import HokuyoSensorSim 

class YoubotComplete:
    def __init__(self):
        # 1. Conexão e Setup
        self.client = RemoteAPIClient()
        self.sim = self.client.getObject('sim')
        
        print("Conectado ao CoppeliaSim.")
        self.client.setStepping(True)
        self.sim.startSimulation()

        # 2. Handles do Braço
        self.arm_joints = []
        for i in range(5):
            self.arm_joints.append(self.sim.getObject(f'/youBot/youBotArmJoint{i}'))
            
        # 3. Handles da Garra
        self.gripper_j1 = self.sim.getObject('/youBot/youBotGripperJoint1')
        self.gripper_j2 = self.sim.getObject('/youBot/youBotGripperJoint2')

        # 4. Handles das Rodas (Mecanum)
        self.wheels = {
            'fl': self.sim.getObject('/youBot/rollingJoint_fl'),
            'rl': self.sim.getObject('/youBot/rollingJoint_rl'),
            'fr': self.sim.getObject('/youBot/rollingJoint_fr'),
            'rr': self.sim.getObject('/youBot/rollingJoint_rr')
        }
        
        # Handle do próprio robô
        self.robot_base = self.sim.getObject('/youBot')

        # 5. Inicialização do Laser
        try:
            self.laser = HokuyoSensorSim(self.sim, '/youBot/fastHokuyo')
        except Exception as e:
            print(f"Aviso Laser: {e}")
            self.laser = None

    def stop(self):
        self.sim.stopSimulation()
        print("Simulação parada.")
    # ==========================================================
    # LÓGICA DO BRAÇO E GARRA (Do main2.py)
    # ==========================================================
    
    def update_gripper_physics(self):
        """Mantém a consistência física da garra."""
        j2_pos = self.sim.getJointPosition(self.gripper_j2)
        self.sim.setJointTargetPosition(self.gripper_j1, j2_pos * -0.5)

    def set_gripper_state(self, action):
        velocity = 0.04
        target = -velocity if action == 'open' else velocity
        self.sim.setJointTargetVelocity(self.gripper_j2, target)

    def wait_action(self, duration):
        """Espera X segundos mantendo a física da garra ativa."""
        dt = self.sim.getSimulationTimeStep()
        steps = int(duration / dt)
        for _ in range(steps):
            self.update_gripper_physics()
            self.client.step()

    def move_arm_smooth(self, target_angles, duration=3.0):
        start_angles = [self.sim.getJointPosition(h) for h in self.arm_joints]
        dt = self.sim.getSimulationTimeStep()
        steps = int(duration / dt)
        
        for i in range(steps):
            t = i / steps
            for j, handle in enumerate(self.arm_joints):
                angle = start_angles[j] + (target_angles[j] - start_angles[j]) * t
                self.sim.setJointTargetPosition(handle, angle)
            
            self.update_gripper_physics()
            self.client.step()
        
        # Posição final exata
        for j, handle in enumerate(self.arm_joints):
            self.sim.setJointTargetPosition(handle, target_angles[j])
        self.client.step()

    # ==========================================================
    # LÓGICA DA BASE MÓVEL
    # ==========================================================

    def set_base_velocity(self, vx, vy, omega):
        """
        Controla as rodas mecanum.
        vx: Velocidade frente/trás (m/s)
        vy: Velocidade lateral (m/s)
        omega: Rotação (rad/s)
        """
        # Constantes geométricas do Youbot
        # Se as rodas girarem errado, inverta os sinais aqui
        # Fórmula Mecanum padrão
        v_fl = -vx + vy + omega 
        v_rl = -vx - vy + omega
        v_fr = -vx - vy - omega
        v_rr = -vx + vy - omega
        
        # Fator de escala para converter m/s linear para rad/s da roda
        k = 20.0 

        self.sim.setJointTargetVelocity(self.wheels['fl'], v_fl * k)
        self.sim.setJointTargetVelocity(self.wheels['rl'], v_rl * k)
        self.sim.setJointTargetVelocity(self.wheels['fr'], v_fr * k)
        self.sim.setJointTargetVelocity(self.wheels['rr'], v_rr * k)

    def stop_base(self):
        self.set_base_velocity(0, 0, 0)
        self.wait_action(0.5) # Estabiliza

    # ==========================================================
    # MODO 1: NAVEGAÇÃO POR WAYPOINT (GLOBAL MAP)
    # ==========================================================

    def go_to_map_position(self, target_x, target_y, tolerance=0.05):
        """
        Navega até uma coordenada (X, Y) do mapa
        """
        print(f"--- Navegando para Waypoint Global: ({target_x}, {target_y}) ---")
        
        while True:
            # 1. Posição e Orientação Global
            current_pos = self.sim.getObjectPosition(self.robot_base, self.sim.handle_world)
            curr_x, curr_y = current_pos[0], current_pos[1]
            
            # Orientação (Yaw/Gamma)
            orientation = self.sim.getObjectOrientation(self.robot_base, self.sim.handle_world)
            curr_theta = orientation[2] # O terceiro ângulo é o Z (Yaw)

            # 2. Erro Global
            dx = target_x - curr_x
            dy = target_y - curr_y
            dist_error = math.sqrt(dx**2 + dy**2)

            if dist_error < tolerance:
                print("Chegou no Waypoint!")
                break

            # 3. Converter Erro Global para Local do Robô (Matriz de Rotação)
            local_x = dx * math.cos(curr_theta) + dy * math.sin(curr_theta)
            local_y = -dx * math.sin(curr_theta) + dy * math.cos(curr_theta)

            # 4. Controle Proporcional 
            kp_lin = 2.0  # Ganho linear
            kp_ang = 4.0  # Ganho angular
            
            # Velocidade simples: Vai direto ajustando X e Y (Holonômico)
            vx = local_x * kp_lin
            vy = local_y * kp_lin
            omega = 0

            # Limitar velocidades máximas
            max_v = 0.5
            vx = max(-max_v, min(max_v, vx))
            vy = max(-max_v, min(max_v, vy))

            self.set_base_velocity(-vx, -vy, -omega)
            
            # Atualiza física da garra e loop
            self.update_gripper_physics()
            self.client.step()
        
        self.stop_base()

    # ==========================================================
    # MODO 2: NAVEGAÇÃO POR LASER
    # ==========================================================

    def approach_object_with_laser(self, target_distance, target_angle_deg, timeout=30):
        """
        Aproximação fina usando laser com alvo específico de ângulo e distância.
        
        Args:
            target_distance (float): Distância alvo em metros (ex: 0.146)
            target_angle_deg (float): Ângulo alvo em graus (ex: -0.7)
        """
        print(f"--- Ajuste Fino: Alvo {target_distance}m / {target_angle_deg}° ---")
        
        # Converter alvo para radianos
        target_angle_rad = math.radians(target_angle_deg)
        start_time = time.time()
        
        while (time.time() - start_time) < timeout:
            # 1. Leitura do Laser
            data = self.laser.getSensorData()
            
            # Se não detectar nada, para e espera (segurança)
            if len(data) == 0:
                self.stop_base()
                self.update_gripper_physics()
                self.client.step()
                continue

            # 2. Filtragem: Pega dados num raio razoável (0.1m a 1.5m)
            ranges = data[:, 1]
            angles = data[:, 0]
            valid_mask = (ranges > 0.05) & (ranges < 1.5)
            
            if not np.any(valid_mask):
                self.stop_base()
                self.update_gripper_physics()
                self.client.step()
                continue
            
            # 3. Identifica o objeto mais próximo
            valid_ranges = ranges[valid_mask]
            valid_angles = angles[valid_mask]
            
            min_idx = np.argmin(valid_ranges)
            curr_dist = valid_ranges[min_idx]
            curr_angle = valid_angles[min_idx]

            # 4. Cálculo dos Erros
            error_dist = curr_dist - target_distance        # Se positivo, precisa ir para frente
            error_angle = curr_angle - target_angle_rad     # Se positivo, precisa girar para direita (sentido horário)

            # 5. Condição de Parada (Tolerância Final)
            # Distância: +/- 2mm (0.002)
            # Ângulo: +/- 0.5 graus (0.008 rad)
            if abs(error_dist) < 0.002 and abs(error_angle) < 0.008:
                print(f"--- ALVO ATINGIDO ---")
                print(f"Dist: {curr_dist:.4f}m (Erro: {error_dist:.4f})")
                print(f"Ang:  {math.degrees(curr_angle):.2f}° (Erro: {math.degrees(error_angle):.2f})")
                break

            # 6. Controlador P
            kp_lin = 0.6   # Ganho linear (mais suave para não bater)
            kp_rot = 0.9   # Ganho rotacional
            
            vx = error_dist * kp_lin
            omega = error_angle * kp_rot
            
            # 7. Saturadores e Velocidade Mínima
            # Se o erro existe mas a velocidade calculada é muito baixa, o robô não anda.
            # Forçamos uma velocidade mínima.
            
            min_v = 0.02
            if abs(vx) < min_v and abs(error_dist) > 0.002:
                vx = math.copysign(min_v, vx)
            
            min_w = 0.05
            if abs(omega) < min_w and abs(error_angle) > 0.008:
                omega = math.copysign(min_w, omega)

            # Teto máximo (Segurança)
            vx = max(-0.15, min(0.15, vx))       # Max 15 cm/s
            omega = max(-0.6, min(0.6, omega)) # Max ~45 deg/s

            # 8. Envia comando
            # Vy = 0 pois vamos girar para alinhar o ângulo, não andar de lado
            self.set_base_velocity(vx, 0, omega)
            
            # Mantém física
            self.update_gripper_physics()
            self.client.step()
        
        self.stop_base()

# ==========================================================
# EXECUÇÃO PRINCIPAL
# ==========================================================
if __name__ == "__main__":
    bot = YoubotComplete()
    
    try:
        # Posição inicial segura do braço
        pose_ready = [0, 0, 0, 0, 0]
        bot.set_gripper_state('open')
        bot.move_arm_smooth(pose_ready, duration=0.5)

        # 1. Aproximação "Grosseira" (Opcional, se o robô estiver longe)
        bot.go_to_map_position(target_x=0.0, target_y=-1.0, tolerance=0.1)
        
        # 2. Aproximação Fina com Laser (Parâmetros exatos)
        # Distância: 0.146m | Ângulo: -0.7 graus
        bot.approach_object_with_laser(target_distance=0.146, target_angle_deg=-0.5)

        print("Posicionamento concluído.")
        # Pequena pausa para estabilizar antes de pegar
        bot.wait_action(0.5)

        # 3. Manipulação (Pegar)
        print("Iniciando pegar objeto...")
        # Pose calculada para pegar algo a 14.6cm da base
        pose_arm = [0, -1.2, -0.7, -1.2, 0] 
        bot.move_arm_smooth(pose_arm, duration=5.0)

        # Fechar Garra
        print("Fechando garra...")
        bot.set_gripper_state('close')
        bot.wait_action(1.0)

        # Posicionar o objeto
        pose_arm = [0.2124,0.7268,0.0,1.6825,0.2224] 
        bot.move_arm_smooth(pose_arm, duration=3.0)
        pose_arm = [0.2124,1.0158,0.0,1.6825,0.2224] 
        bot.move_arm_smooth(pose_arm, duration=3.0)
        # Soltar o objeto
        print("Abrindo garra...")
        bot.set_gripper_state('open')
        bot.wait_action(1.0) 
        # Voltar para posição neutra
        pose_arm = [0, 0, 0, 0, 0] 
        bot.move_arm_smooth(pose_arm, duration=3.0)

        # Transportar o objeto para outro local
        bot.go_to_map_position(target_x=1.0, target_y=-1.0, tolerance=0.1)

        # Pegar o objeto de volta
        pose_arm = [0.2124,0.7268,0.0,1.6825,0.2224] 
        bot.move_arm_smooth(pose_arm, duration=2.0)
        pose_arm = [0.2124,1.0158,0.0,1.6825,0.2224] 
        bot.move_arm_smooth(pose_arm, duration=2.0)
        print("Fechando garra...")
        bot.set_gripper_state('close')
        bot.wait_action(1.0) 

        pose_arm = [0.2124,0.7268,0.0,1.6825,0.2224] 
        bot.move_arm_smooth(pose_arm, duration=2.0)
        pose_arm = [1.5,0.7268,0.0,1.6825,0.2224] 
        bot.move_arm_smooth(pose_arm, duration=2.0)
        pose_arm = [1.5, 1, 0.8, 1.2, 0]
        bot.move_arm_smooth(pose_arm, duration=3.0)
        # Soltar o objeto
        print("Abrindo garra...")
        bot.set_gripper_state('open')
        bot.wait_action(1.0) 

        pose_arm = [0, 0, 0, 0, 0]
        bot.move_arm_smooth(pose_arm, duration=2.0)
        bot.go_to_map_position(target_x=-1.0, target_y=-1.0, tolerance=0.1)
        
        
        bot.wait_action(2.0) # Fim 


    except KeyboardInterrupt:
        print("Parando...")
    except Exception as e:
        print(f"Erro: {e}")
    finally:
        bot.stop()