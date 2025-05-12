"""
Servo control classes for face tracking system.
Provides Servos, PID, and Gimbal classes for controlling servo motors.
"""
from maix import pwm

class Servos:
    """
    Control a servo using PWM.
    """
    def __init__(self, pin_name, init_pos, duty_min, duty_max):
        """
        Initialize a servo controller.
        
        Args:
            pin_name (str): PWM pin name
            init_pos (float): Initial position (0-100)
            duty_min (float): Minimum duty cycle
            duty_max (float): Maximum duty cycle
        """
        self.pin_name = pin_name
        self.duty_min = duty_min
        self.duty_max = duty_max
        self.position = init_pos
        self.duty_range = duty_max - duty_min
        
        # Initialize PWM
        try:
            self.pwm_handle = pwm.PWM(pin_name)
            self.pwm_handle.period(20000)  # 20ms (50Hz)
            self.duty = self.duty_min + (self.position / 100.0) * self.duty_range
            self.pwm_handle.duty(self.duty)
            self.enable()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize servo on {pin_name}: {e}")
            
    def enable(self):
        """Enable the servo."""
        self.pwm_handle.enable()
        
    def disable(self):
        """Disable the servo."""
        self.pwm_handle.disable()
        
    def set_pos(self, pos):
        """
        Set the servo position.
        
        Args:
            pos (float): Position value (0-100)
        """
        if pos < 0:
            pos = 0
        elif pos > 100:
            pos = 100
            
        self.position = pos
        self.duty = self.duty_min + (pos / 100.0) * self.duty_range
        self.pwm_handle.duty(self.duty)
        
    def get_pos(self):
        """Get the current servo position."""
        return self.position
        
    def cleanup(self):
        """Clean up resources."""
        self.disable()


class PID:
    """
    PID controller for smooth servo control.
    """
    def __init__(self, p=0.1, i=0.0, d=0.0, imax=0):
        """
        Initialize a PID controller.
        
        Args:
            p (float): Proportional gain
            i (float): Integral gain
            d (float): Derivative gain
            imax (float): Maximum integral value
        """
        self.p = p
        self.i = i
        self.d = d
        self.imax = imax
        self.reset()
        
    def reset(self):
        """Reset PID values."""
        self.integral = 0
        self.prev_err = 0
        
    def calc(self, err):
        """
        Calculate PID output.
        
        Args:
            err (float): Error value
            
        Returns:
            float: PID output
        """
        # Proportional
        p_out = self.p * err
        
        # Integral
        self.integral += err
        if self.imax > 0 and abs(self.integral) > self.imax:
            self.integral = self.imax if self.integral > 0 else -self.imax
        i_out = self.i * self.integral
        
        # Derivative
        derivative = err - self.prev_err
        self.prev_err = err
        d_out = self.d * derivative
        
        # Sum all components
        return p_out + i_out + d_out


class Gimbal:
    """
    Control a gimbal using two servos.
    """
    def __init__(self, pitch_servo, pitch_pid, roll_servo, roll_pid):
        """
        Initialize a gimbal controller.
        
        Args:
            pitch_servo (Servos): Pitch servo controller
            pitch_pid (PID): Pitch PID controller
            roll_servo (Servos): Roll servo controller
            roll_pid (PID): Roll PID controller
        """
        self.pitch = pitch_servo
        self.roll = roll_servo
        self.pitch_pid = pitch_pid
        self.roll_pid = roll_pid
        
    def run(self, err_pitch, err_roll, pitch_reverse=False, roll_reverse=False):
        """
        Run the gimbal with given error values.
        
        Args:
            err_pitch (float): Pitch error value
            err_roll (float): Roll error value
            pitch_reverse (bool): Reverse pitch direction
            roll_reverse (bool): Reverse roll direction
        """
        # Calculate PID outputs
        pitch_out = self.pitch_pid.calc(err_pitch)
        roll_out = self.roll_pid.calc(err_roll)
        
        # Apply direction reverse if needed
        if pitch_reverse:
            pitch_out = -pitch_out
        if roll_reverse:
            roll_out = -roll_out
        
        # Update servo positions
        cur_pitch = self.pitch.get_pos()
        cur_roll = self.roll.get_pos()
        
        new_pitch = cur_pitch + pitch_out
        new_roll = cur_roll + roll_out
        
        # Set new positions
        self.pitch.set_pos(new_pitch)
        self.roll.set_pos(new_roll)
        
    def cleanup(self):
        """Clean up resources."""
        self.pitch.cleanup()
        self.roll.cleanup() 