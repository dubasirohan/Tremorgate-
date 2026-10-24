import csv, time, argparse, serial

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",required=True,help="CSV with 12 columns matching TremorGate contract")
    ap.add_argument("--port",default="/tmp/ttyV0")
    ap.add_argument("--baud",type=int,default=115200)
    ap.add_argument("--rate",type=float,default=100.0)
    ap.add_argument("--loop",action="store_true")
    args=ap.parse_args()
    while True:
        with serial.Serial(args.port,args.baud,timeout=1) as ser, open(args.input,newline="") as f:
            rows=csv.reader(f)
            for row in rows:
                if not row or row[0].startswith("#"): continue
                if len(row)!=12: continue
                ser.write((",".join(row)+"\n").encode())
                time.sleep(1.0/args.rate)
        if not args.loop: break

if __name__=="__main__": main()
