import Serial
import time


if __name__ == '__main__':
   ser = serial.Serial('/dev/ttyAMC0',9600, timeout = 1)
   ser.reset_input_buffer()

   while True:
       ser.write(b"F/n")
       line = se.readline().decode('utf-8').rstrip()
       print(line)
       time.sleep(1)
