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
        v_fl, v_rl = -vx + vy + omega, -vx - vy + omega
        v_fr, v_rr = -vx - vy - omega, -vx + vy - omega
        k = 20.0
        self.sim.setJointTargetVelocity(self.wheels['fl'], v_fl * k)
        self.sim.setJointTargetVelocity(self.wheels['rl'], v_rl * k)
        self.sim.setJointTargetVelocity(self.wheels['fr'], v_fr * k)
        self.sim.setJointTargetVelocity(self.wheels['rr'], v_rr * k)

    def stop_base(self):
        self.set_base_velocity(0, 0, 0)
        self.wait_action(0.5)

    def go_to_map_position(self, target_x, target_y, tolerance=0.05):
        print(f"-> Indo para: ({target_x}, {target_y})")
        while True:
            curr_pos = self.sim.getObjectPosition(self.robot_base, self.sim.handle_world)
            curr_theta = self.sim.getObjectOrientation(self.robot_base, self.sim.handle_world)[0]
            dx, dy = target_x - curr_pos[0], target_y - curr_pos[1]
            if math.sqrt(dx**2 + dy**2) < tolerance: break
            
            local_x = dx * math.cos(curr_theta) + dy * math.sin(curr_theta)
            local_y = -dx * math.sin(curr_theta) + dy * math.cos(curr_theta)
            vx, vy = max(-0.5, min(0.5, local_x*2.0)), max(-0.5, min(0.5, local_y*2.0))
            self.set_base_velocity(-vx, -vy, 0)
            self.update_gripper_physics()
            self.client.step()
        self.stop_base()

    def approach_object_with_laser(self, target_dist, target_ang_deg, timeout=30):
        print(f"-> Laser Fino: {target_dist}m / {target_ang_deg}°")
        target_ang_rad = math.radians(target_ang_deg)
        start = time.time()
        while (time.time() - start) < timeout:
            data = self.laser.getSensorData()
            if len(data) == 0: 
                self.stop_base()
                self.update_gripper_physics()
                self.client.step()
                continue
            
            ranges = data[:, 1]
            mask = (ranges > 0.05) & (ranges < 1.5)
            if not np.any(mask): continue
            
            idx = np.argmin(ranges[mask])
            c_dist = ranges[mask][idx]
            c_ang = data[:, 0][mask][idx]
            
            e_dist, e_ang = c_dist - target_dist, c_ang - target_ang_rad
            if abs(e_dist) < 0.002 and abs(e_ang) < 0.008: break
            
            vx = max(-0.15, min(0.15, e_dist * 0.6))
            if abs(vx) < 0.02 and abs(e_dist) > 0.002: vx = math.copysign(0.02, vx)
            
            omega = max(-0.6, min(0.6, e_ang * 0.9))
            if abs(omega) < 0.05 and abs(e_ang) > 0.008: omega = math.copysign(0.05, omega)

            self.set_base_velocity(vx, 0, omega)
            self.update_gripper_physics()
            self.client.step()
        self.stop_base()

# ==========================================================
# CONFIGURAÇÃO DE POSIÇÕES (CORRIGIDA)
# ==========================================================
# High: Posição intermediária (mais alta)
# Low: Posição final de encaixe (mais baixa na junta 1)
STORAGE_POSES = [
     # OBJETO 3 (main copy 1)
    {
        "id": 3,
        "pose_high": [-0.248, 0.2741, 0.4516, 1.7806, -0.199],
        "pose_low":  [-0.248, 0.7637, 0.4516, 1.7806, -0.199]
    },
    # OBJETO 1 (main copy 2)
    {
        "id": 1,
        "pose_high": [0.3544, 0.2515, 0.4516, 1.7806, 0.4531], 
        "pose_low":  [0.3544, 0.7435, 0.4516, 1.7806, 0.4531]
    },
        # OBJETO 4 (main copy 3)
    {
        "id": 4,
        "pose_high": [-0.2141, 0.6107, 0.0, 1.7308, -0.2493],
        "pose_low":  [-0.2141, 1.0,    0.0, 1.7308, -0.2493]
    },
    # OBJETO 2 (main copy 4)
    {
        "id": 2,
        "pose_high": [0.2124, 0.7268, 0.0, 1.6825, 0.2224], 
        "pose_low":  [0.2124, 1.0158, 0.0, 1.6825, 0.2224]  
    }

]

POSE_NEUTRAL = [0, 0, 0, 0, 0]
POSE_PICK_GROUND = [0, -1.2, -0.7, -1.2, 0]
POSE_DROP_SIDE = [1.5, 1, 0.8, 1.2, 0] 

# ==========================================================
# SEQUÊNCIAS LÓGICAS
# ==========================================================

def execute_full_cycle(bot):
    bot.go_to_map_position(1.0, 0, tolerance=0.1)
    # -------------------------------------------------
    # FASE 1: COLETA (1 -> 2 -> 3 -> 4)
    # -------------------------------------------------
    for obj in STORAGE_POSES:
        print(f"\n=== [COLETA] Objeto {obj['id']} ===")
        
        # 1. Navegar
        
        # 2. Preparar braço e Laser
        bot.set_gripper_state('open')
        bot.move_arm_smooth(POSE_NEUTRAL, duration=1.0)
        bot.approach_object_with_laser(0.146, -0.5)
        bot.wait_action(0.5)

        # 3. Pegar do chão
        print("Pegando...")
        bot.move_arm_smooth(POSE_PICK_GROUND, duration=4.0)
        bot.set_gripper_state('close')
        bot.wait_action(1.0)

        # 4. Guardar na Base (Sequência Crítica)
        print("Guardando na base...")
        # A. Levanta um pouco do chão (segurança)
        pose_safe = list(POSE_PICK_GROUND)
        pose_safe[1] = -0.6
        bot.move_arm_smooth(pose_safe, duration=2.0)

        # B. Vai para posição ALTA (High)
        bot.move_arm_smooth(obj['pose_high'], duration=3.0)

        # C. Vai para posição BAIXA/ENCAIXE (Low)
        bot.move_arm_smooth(obj['pose_low'], duration=3.0)
        
        # D. Pausa para estabilizar ANTES de soltar
        bot.wait_action(0.5) 
        
        # E. Solta
        bot.set_gripper_state('open')
        bot.wait_action(1.0) # Espera a garra abrir totalmente

        # F. Volta para posição ALTA (High) para não arrastar no objeto
        bot.move_arm_smooth(obj['pose_high'], duration=2.0)
        
        # G. Neutro
        bot.move_arm_smooth(POSE_NEUTRAL, duration=2.0)

    # -------------------------------------------------
    # FASE 2: TRANSPORTE
    # -------------------------------------------------
    print("\n=== TRANSPORTANDO ===")
    bot.go_to_map_position(-2, -2.0, tolerance=0.1)

    # -------------------------------------------------
    # FASE 3: ENTREGA (4 -> 3 -> 2 -> 1)
    # -------------------------------------------------
    # Inverte a lista para tirar o último primeiro
    for obj in reversed(STORAGE_POSES):
        print(f"\n=== [ENTREGA] Objeto {obj['id']} ===")
        
        bot.set_gripper_state('open')
        
        # 1. Pegar da Base
        # A. Posição ALTA
        bot.move_arm_smooth(obj['pose_high'], duration=2.0)
        
        # B. Posição BAIXA (Onde o objeto está)
        bot.move_arm_smooth(obj['pose_low'], duration=2.0)
        
        # C. Fechar Garra
        bot.set_gripper_state('close')
        bot.wait_action(1.0) # Espera firmar a pega

        # D. Levanta para posição ALTA
        bot.move_arm_smooth(obj['pose_high'], duration=2.0)

        # 2. Jogar no Chão
        print("Depositando no destino...")
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