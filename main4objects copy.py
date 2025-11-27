import time
import math
import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from hokuyo import HokuyoSensorSim 

class YoubotComplete:
    def __init__(self):
        self.client = RemoteAPIClient()
        self.sim = self.client.getObject('sim')
        print("Conectado ao CoppeliaSim.")
        self.client.setStepping(True)
        self.sim.startSimulation()

        self.arm_joints = [self.sim.getObject(f'/youBot/youBotArmJoint{i}') for i in range(5)]
        self.gripper_j1 = self.sim.getObject('/youBot/youBotGripperJoint1')
        self.gripper_j2 = self.sim.getObject('/youBot/youBotGripperJoint2')
        self.wheels = {
            'fl': self.sim.getObject('/youBot/rollingJoint_fl'),
            'rl': self.sim.getObject('/youBot/rollingJoint_rl'),
            'fr': self.sim.getObject('/youBot/rollingJoint_fr'),
            'rr': self.sim.getObject('/youBot/rollingJoint_rr')
        }
        self.robot_base = self.sim.getObject('/youBot')
        try:
            self.laser = HokuyoSensorSim(self.sim, '/youBot/fastHokuyo')
        except:
            self.laser = None

    def stop(self):
        self.sim.stopSimulation()
        print("Simulação parada.")

    # --- FÍSICA E BRAÇO ---
    def update_gripper_physics(self):
        j2_pos = self.sim.getJointPosition(self.gripper_j2)
        self.sim.setJointTargetPosition(self.gripper_j1, j2_pos * -0.5)

    def set_gripper_state(self, action):
        velocity = 0.04
        target = -velocity if action == 'open' else velocity
        self.sim.setJointTargetVelocity(self.gripper_j2, target)

    def wait_action(self, duration):
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
        for j, handle in enumerate(self.arm_joints):
            self.sim.setJointTargetPosition(handle, target_angles[j])
        self.client.step()

    # --- BASE E NAVEGAÇÃO ---
    def set_base_velocity(self, vx, vy, omega):
        # Fórmula Mecanum Simples
        v_fl = -vx + vy + omega
        v_rl = -vx - vy + omega
        v_fr = -vx - vy - omega
        v_rr = -vx + vy - omega
        
        k = 20.0
        self.sim.setJointTargetVelocity(self.wheels['fl'], v_fl * k)
        self.sim.setJointTargetVelocity(self.wheels['rl'], v_rl * k)
        self.sim.setJointTargetVelocity(self.wheels['fr'], v_fr * k)
        self.sim.setJointTargetVelocity(self.wheels['rr'], v_rr * k)

    def stop_base(self):
        self.set_base_velocity(0, 0, 0)
        self.wait_action(0.5)

    def normalize_angle(self, angle):
        while angle > math.pi: angle -= 2 * math.pi
        while angle < -math.pi: angle += 2 * math.pi
        return angle

    def rotate_to_angle(self, target_angle_rad, tolerance=0.05):
        """
        Gira o robô no próprio eixo até atingir o ângulo global alvo.
        Usado para 'zerar' a orientação antes de viagens longas.
        """
        print(f"--- Realinhando para {math.degrees(target_angle_rad):.1f}° ---")
        while True:
            # Pega orientação atual
            curr_theta = self.sim.getObjectOrientation(self.robot_base, self.sim.handle_world)[0]
            diff = self.normalize_angle(target_angle_rad - curr_theta)
            print(abs(diff))
            if abs(diff) < tolerance:
                break
            
            # Controle P simples apenas para rotação (sem andar)
            omega = diff * 1.5
            omega = max(-0.8, min(0.8, omega)) # Limita velocidade
            
            # vx=0, vy=0, apenas roda
            self.set_base_velocity(0, 0, -omega) # Sinal invertido conforme sua calibração
            
            self.update_gripper_physics()
            self.client.step()
        
        self.stop_base()

    def go_to_map_position(self, target_x, target_y, tolerance=0.05):
        """
        Versão clássica/estável sem correção ativa de ângulo (evita oscilação).
        """
        print(f"-> Navegando para: ({target_x}, {target_y})")
        while True:
            curr_pos = self.sim.getObjectPosition(self.robot_base, self.sim.handle_world)
            curr_theta = self.sim.getObjectOrientation(self.robot_base, self.sim.handle_world)[0]
            
            dx = target_x - curr_pos[0]
            dy = target_y - curr_pos[1]
            dist_error = math.sqrt(dx**2 + dy**2)
            
            if dist_error < tolerance:
                break
            
            # Converte erro global para local
            local_x = dx * math.cos(curr_theta) + dy * math.sin(curr_theta)
            local_y = -dx * math.sin(curr_theta) + dy * math.cos(curr_theta)
            
            kp = 2.0
            vx = local_x * kp
            vy = local_y * kp
            
            max_v = 0.5
            vx = max(-max_v, min(max_v, vx))
            vy = max(-max_v, min(max_v, vy))

            self.set_base_velocity(-vx, -vy, 0) # Sem rotação (omega=0)
            
            self.update_gripper_physics()
            self.client.step()
        self.stop_base()

    # --- LASER E BUSCA 360 ---
    def _check_laser(self):
        data = self.laser.getSensorData()
        if len(data) == 0: return False, None, None
        ranges = data[:, 1]
        mask = (ranges > 0.05) & (ranges < 1.5)
        if not np.any(mask): return False, None, None
        return True, data, mask

    def perform_360_search(self):
        print("--- Varredura 360º Iniciada ---")
        # Gira uma volta completa
        omega_search = 0.5
        duration = (2 * math.pi / omega_search) + 1.0 
        start_time = time.time()
        found = False
        
        while (time.time() - start_time) < duration:
            self.set_base_velocity(0, 0, -omega_search)
            self.update_gripper_physics()
            
            # Se viu algo, para imediatamente
            if self._check_laser()[0]:
                print("!!! Objeto detectado na varredura !!!")
                found = True
                break
            self.client.step()
            
        self.stop_base()
        return found

    def approach_object_with_laser(self, target_dist, target_ang_deg, timeout=30):
        print(f"-> Laser Fino: {target_dist}m / {target_ang_deg}°")
        target_ang_rad = math.radians(target_ang_deg)
        
        # 1. Verifica se vê o objeto logo de cara
        seen, _, _ = self._check_laser()
        
        # 2. Se não viu, faz a busca 360
        if not seen:
            found = self.perform_360_search()
            if not found:
                print("Nada encontrado após 360º.")
                # Opcional: Aqui poderíamos voltar a orientação original se falhar,
                # mas o ciclo principal vai forçar o realinhamento de qualquer jeito.
                return 

        # 3. Aproximação fina (PID)
        start = time.time()
        while (time.time() - start) < timeout:
            seen, data, mask = self._check_laser()
            if not seen:
                self.stop_base()
                self.update_gripper_physics()
                self.client.step()
                continue
            
            idx = np.argmin(data[:, 1][mask])
            c_dist = data[:, 1][mask][idx]
            c_ang = data[:, 0][mask][idx]
            
            e_dist = c_dist - target_dist
            e_ang = c_ang - target_ang_rad
            
            if abs(e_dist) < 0.002 and abs(e_ang) < 0.008: 
                print("Alvo Laser OK")
                break
            
            vx = max(-0.15, min(0.15, e_dist * 0.6))
            if abs(vx) < 0.02 and abs(e_dist) > 0.002: vx = math.copysign(0.02, vx)
            
            omega = max(-0.6, min(0.6, e_ang * 0.9))
            if abs(omega) < 0.05 and abs(e_ang) > 0.008: omega = math.copysign(0.05, omega)

            self.set_base_velocity(vx, 0, omega) # Omega positivo aqui funcionava no original
            self.update_gripper_physics()
            self.client.step()
        self.stop_base()

# ==========================================================
# EXECUÇÃO
# ==========================================================
STORAGE_POSES = [
    {"id": 1, "pose_high": [0.3544, 0.2515, 0.4516, 1.7806, 0.4531], "pose_low": [0.3544, 0.7435, 0.4516, 1.7806, 0.4531]},
    {"id": 2, "pose_high": [0.2124, 0.7268, 0.0, 1.6825, 0.2224], "pose_low": [0.2124, 1.0158, 0.0, 1.6825, 0.2224]},
    {"id": 3, "pose_high": [-0.248, 0.2741, 0.4516, 1.7806, -0.199], "pose_low": [-0.248, 0.7637, 0.4516, 1.7806, -0.199]},
    {"id": 4, "pose_high": [-0.2141, 0.6107, 0.0, 1.7308, -0.2493], "pose_low": [-0.2141, 1.0, 0.0, 1.7308, -0.2493]}
]

POSE_NEUTRAL = [0, 0, 0, 0, 0]
POSE_PICK_GROUND = [0, -1.2, -0.7, -1.2, 0]
POSE_DROP_SIDE = [1.5, 1, 0.8, 1.2, 0] 

def execute_full_cycle(bot):
    # FASE 1: COLETA
    bot.go_to_map_position(1.0, 0, tolerance=0.1)
    for obj in STORAGE_POSES:
        print(f"\n=== [COLETA] Objeto {obj['id']} ===")
        
        
        bot.set_gripper_state('open')
        bot.move_arm_smooth(POSE_NEUTRAL, duration=1.0)
        
        # Tenta achar e alinhar
        bot.approach_object_with_laser(0.146, -0.5) 
        bot.wait_action(0.5)

        print("Pegando...")
        bot.move_arm_smooth(POSE_PICK_GROUND, duration=4.0)
        bot.set_gripper_state('close')
        bot.wait_action(1.0)

        print("Guardando na base...")
        pose_safe = list(POSE_PICK_GROUND)
        pose_safe[1] = -0.6 # Levanta ombro
        bot.move_arm_smooth(pose_safe, duration=2.0)
        
        # Deposita
        bot.move_arm_smooth(obj['pose_high'], duration=3.0)
        bot.move_arm_smooth(obj['pose_low'], duration=3.0)
        bot.wait_action(0.5) 
        bot.set_gripper_state('open')
        bot.wait_action(1.0)
        
        # Retira braço
        bot.move_arm_smooth(obj['pose_high'], duration=2.0)
        bot.move_arm_smooth(POSE_NEUTRAL, duration=2.0)

        # === REALINHAMENTO CRÍTICO ===
        # Após a bagunça do laser/busca e manipulação,
        # garantimos que o robô olhe para frente (0 graus)
        # antes de fazer qualquer outro movimento longo.
        bot.rotate_to_angle(0.0) 

    # FASE 2: TRANSPORTE
    print("\n=== TRANSPORTANDO ===")
    # Como já alinhamos no final do loop acima, o robô deve ir reto
    bot.go_to_map_position(1.5, -1.0, tolerance=0.1)

    # FASE 3: ENTREGA
    # Opcional: Realinhar ao chegar, para garantir precisão no drop
    bot.rotate_to_angle(0.0)

    for obj in reversed(STORAGE_POSES):
        print(f"\n=== [ENTREGA] Objeto {obj['id']} ===")
        bot.set_gripper_state('open')
        bot.move_arm_smooth(obj['pose_high'], duration=2.0)
        bot.move_arm_smooth(obj['pose_low'], duration=2.0)
        bot.set_gripper_state('close')
        bot.wait_action(1.0)
        bot.move_arm_smooth(obj['pose_high'], duration=2.0)
        
        print("Depositando no chão...")
        bot.move_arm_smooth(POSE_DROP_SIDE, duration=3.0)
        bot.set_gripper_state('open')
        bot.wait_action(1.0)
        bot.move_arm_smooth(POSE_NEUTRAL, duration=2.0)

if __name__ == "__main__":
    bot = YoubotComplete()
    try:
        execute_full_cycle(bot)
        print("Finalizado com sucesso.")
    except KeyboardInterrupt:
        print("Parando...")
    except Exception as e:
        print(f"Erro: {e}")
    finally:
        bot.stop()