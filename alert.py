import subprocess

def send_alert(title, message, urgency, time):
	subprocess.run(["notify-send", "-u", urgency, "-t", str(time), title, message])

def emit_test_alert():
	send_alert("Test", "This is a test alert", "critical", 5000)

if __name__ == "__main__":
	emit_test_alert()