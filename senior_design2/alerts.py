import sys
import sx126x
import threading
import time
import select
import termios
import tty
from threading import Timer

old_settings = termios.tcgetattr(sys.stdin)
tty.setcbreak(sys.stdin.fileno())

node = sx126x.sx126x(serial_num = "/dev/ttyS0",freq=915,addr=0,power=22,rssi=True,air_speed=2400,relay=False)

def get_cpu_temp():
    tempFile = open( "/sys/class/thermal/thermal_zone0/temp" )
    cpu_temp = tempFile.read()
    tempFile.close()
    return float(cpu_temp)/1000

def send_cpu_continue(continue_or_not = True):
    if continue_or_not:
        if get_cpu_temp() > 50.0:   #Only transmit if the temperature is above threshold
                global timer_task
                global seconds
        
        # broadcast the cpu temperature at 915MHz
                data = bytes([255]) + bytes([255]) + bytes([18]) + bytes([255]) + bytes([255]) + bytes([12]) + " CPU Temperature Warning:".encode()+str(get_cpu_temp()).encode()+" C".encode()
                node.send(data)
                time.sleep(0.2)

        timer_task = Timer(seconds,send_cpu_continue)
        timer_task.start()
    else:
        data = bytes([255]) + bytes([255]) + bytes([18]) + bytes([255]) + bytes([255]) + bytes([12]) + " CPU Temperature Warning:".encode()+str(get_cpu_temp()).encode()+" C".encode()
        node.send(data)
        time.sleep(0.2)
        timer_task.cancel()
        pass

def send_low_battery():
    data2 = bytes([255]) + bytes([255]) + bytes([18]) + bytes([255]) + bytes([255]) + bytes([12]) + " Low Battery Warning".encode()
    node.send(data2)
    time.sleep(0.2)
    timer_lowBattery.cancel()

try:
# it will check rpi cpu temperature every 30 seconds 
    seconds = 30
    #Low battery alert after 50 min of run time
    battLife = 3000

    timer_task = Timer(seconds,send_cpu_continue)
    timer_task.start()
    timer_lowBattery = Timer(battLife, send_low_battery)
    timer_lowBattery.start()
    
except:
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
    # print('\x1b[2A',end='\r')
    # print(" "*100)
    # print(" "*100)
    # print('\x1b[2A',end='\r')

termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
# print('\x1b[2A',end='\r')
# print(" "*100)
# print(" "*100)
# print('\x1b[2A',end='\r')
