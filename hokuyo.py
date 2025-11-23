import numpy as np
import math
import time

# ==================== HOKUYO SENSOR CLASS ====================
class HokuyoSensorSim(object):
    """
    Simulates a Hokuyo laser sensor in CoppeliaSim using vision sensors.

    This class provides an interface to interact with a simulated Hokuyo sensor,
    typically attached to a robot in CoppeliaSim. It manages the underlying vision
    sensors and provides a method to retrieve sensor data in range format.

    Attributes:
        _sim: The simulation API object used to interact with CoppeliaSim.
        _base_name (str): The name of the base object to which the Hokuyo sensor is attached.
        _is_range_data (bool): Determines if sensor data is returned as range values (True).
        _base_obj: The handle of the base object in the simulation.
        _vision_sensors_obj (list): Handles of the vision sensors used to simulate the Hokuyo sensor.
        _angles_lua (np.array): Stores the precise angles read from the simulation.
        _is_initialized_angles (bool): Indicates if the angles were read successfully.
    
    Args:
        sim: The simulation API object.
        base_name (str): The name of the base object (must contain 'fastHokuyo').
        is_range_data (bool, optional): If True, sensor data is returned as range values. Defaults to True.
        
    Raises:
        ValueError: If 'fastHokuyo' is not in the base_name, or if the base object or vision sensors are not found.

    Methods:
        get_is_range_data() -> bool:
            Returns whether sensor data is returned as range values.

        set_is_range_data(is_range_data: bool) -> None:
            Sets whether sensor data should be returned as range values.

        getSensorData():
            Retrieves sensor data from the vision sensors.
            Returns a list of [angle, range] values.
    """
    
    _sim = None
    
    _base_name = "" 

    _angles_lua = np.empty(0) 
    _is_initialized_angles = False 
    ANGLE_SIGNAL = 'signal.hokuyo_angle_data' 
    
    _base_obj = None 
    _is_range_data = True
    _vision_sensors_obj = []

    def __init__(self, sim, base_name, is_range_data=True):
        self._sim = sim
        self.base_name = base_name
        self._is_range_data = is_range_data

        if "fastHokuyo" not in base_name:
            raise ValueError(
                f"ERR: fastHokuyo must be in the base object name. Ex: `/kobuki/fastHokuyo`"
            )

        self._base_obj = sim.getObject(base_name)
        if self._base_obj == -1:
            raise ValueError(
                f"ERR: base_obj ({self._base_obj}) is not a valid name in the simulation"
            )

        self._vision_sensors_obj = [
            sim.getObject(f'{base_name}/fastHokuyo_sensor1'),
            sim.getObject(f'{base_name}/fastHokuyo_sensor2'),
        ]

        if any(obj == -1 for obj in self._vision_sensors_obj):
            raise ValueError(
                f"ERR: the _vision_sensors_obj names are not valid in the simulation"
            )

    def get_is_range_data(self) -> bool:
        return self._is_range_data

    def set_is_range_data(self, is_range_data: bool) -> None:
        self._is_range_data = is_range_data

    def _initialize_angles_from_lua(self):

        # Essas variáveis agora são utilizadas apenas quando os valores do laser 
        # não são desempacotados diretamente do sensor, como um fallback em caso de falha
        angle_min = -120 * math.pi / 180
        angle_increment = (240 / 684) * math.pi / 180
        total_steps = 684
        
        for i in range(15):
            try:
                if self._vision_sensors_obj and self._vision_sensors_obj[0] != -1:
                    self._sim.readVisionSensor(self._vision_sensors_obj[0]) 

                angles_packed = self._sim.getBufferProperty(self._sim.handle_scene, self.ANGLE_SIGNAL, {'noError' : True})
                
                if angles_packed:
                    self._angles_lua = np.array(self._sim.unpackFloatTable(angles_packed))
                    self._is_initialized_angles = True
                    print(f"Precise sensor angles read on attempt {i+1} ({self._angles_lua.size} laser beam readings)")
                    return True

                self._sim.step()
                time.sleep(0.01)

            except Exception as e:
                pass 
                
        self._angles_lua = np.arange(angle_min, angle_min + total_steps * angle_increment, angle_increment)
        self._is_initialized_angles = True
        print("ALERT: Communication via buffer property failed. Using approximation. Map may be blurred.")
        return False

    def getSensorData(self):
        
        if not self._is_initialized_angles:
            self._initialize_angles_from_lua()
        
        sensor_data = []
        angle_idx = 0
        angles_to_use = self._angles_lua
        
        if angles_to_use.size == 0:
            return np.empty((0, 2))
        
        for vision_sensor in self._vision_sensors_obj:
            
            result = self._sim.readVisionSensor(vision_sensor)
            if not isinstance(result, (list, tuple)) or len(result) != 3: continue 
                
            r, t, u = result
            if u:
                for j in range(int(u[1])): 
                    for k in range(int(u[0])): 
                        w_idx = 2 + 4 * (j * int(u[0]) + k)
                        v_dist = u[w_idx + 3] 
                        if angle_idx < angles_to_use.size:
                            current_angle = angles_to_use[angle_idx]
                            sensor_data.append([current_angle, v_dist])
                            angle_idx += 1
                        else:
                            break 
                    if angle_idx >= angles_to_use.size: break

        return np.array(sensor_data) if sensor_data else np.empty((0, 2))