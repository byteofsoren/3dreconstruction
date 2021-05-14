class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OK = '\033[92m'
    WARN = '\033[93m'
    ERR = '\033[38;5;9m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UND = '\033[4m'
    INF = '\033[38;5;81m'
    class SIGN:
        OK = "\033[92m[OK]\033[0m"
        NO = "\033[93m[NO]\033[0m"
        FAIL = "\033[91m[FAIL]\033[0m"
        FIXED  = "\033[96m[FIXED]\033[0m"
    def tfcolor(t):
        if t:
            return "\033[94m[True]\033[0m"
        else:
            return "\033[38;5;202m[False]\033[0m"


