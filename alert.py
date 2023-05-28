import subprocess
import requests
import json
from time import sleep

def send_alert(title, message, urgency, time, sound=False, discord=False):
	subprocess.run(["xset", "dpms", "force", "on"])
	subprocess.run(["notify-send", "-u", urgency, "-t", str(time), title, message]) # possible urgency values for notify-send: low, normal, critical
	if discord:
		data = {
			"content": f"!!!!!!!!!! **{title}** !!!!!!!!!!\n{message}",
			"username": "ACARISTrainer"
		}

		response = requests.post("https://discord.com/api/webhooks/1109919684050034719/E0Nz1r5qXAsRP_EJfAGaJmRPXS_fnJLg3tCsxpcjssZUBC2JWdTP-zpbD3o2tmQShFvy", json=data)
	if sound: # beep using sound that lasts 8 second until killed
		while True:
			subprocess.run(["aplay", "-q", "/home/simtoon/Documents/alarm.wav"])
			sleep(8)


def emit_test_alert():
	send_alert("Test", "This is a test alert", "critical", 5000, sound=True, discord=True)

if __name__ == "__main__":
	emit_test_alert()