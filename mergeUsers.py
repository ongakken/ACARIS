import csv

smtnUsernames = {
	"motoko#0001": "simtoon",
	"motoko#9418": "simtoon",
	"TheKentuckian#0001": "simtoon",
	"DrippingPussyUS#0001": "simtoon",
	"PrincessAhegao#0001": "simtoon",
	"Princess4h3g4oUS#0001": "simtoon",
	"TheKentuckian#6304": "simtoon",
	"J3ff#0001": "simtoon",
	"DerFeuchteGigachad#8218": "simtoon"
}

with open("sents.csv", "r") as f_in, open("sents_merged.csv", "w", newline="") as f_out:
	reader = csv.reader(f_in, delimiter=",")
	writer = csv.writer(f_out, delimiter=",")
	for row in reader:
		userID = row[0]
		if userID in smtnUsernames:
			userID = smtnUsernames[userID]
		writer.writerow([userID] + row[1:])