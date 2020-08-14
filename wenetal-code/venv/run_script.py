import os, sys

print(sys.argv)

try:
    assert(len(sys.argv) <= 3)
except AssertionError:
    print("At most three arguments allowed.")
    print("e.g. python3 dubins_run.py dubins_naf.py")
    print("     this will run the specified file for 10 times.")
    print("or:  python3 dubins_run.py dubins_naf.py 3")
    print("     the third argument is the number of times to run the file.")
    sys.exit(-1)

if len(sys.argv) == 3:
    try:
        repeat_times = int(sys.argv[2])
    except ValueError:
        print("Make sure that the third argument is a positive integer.")
        sys.exit(-2)
else:
     repeat_times = 10

try:
    f = open(sys.argv[1], 'r')
    f.close()
except ValueError:
    print("Please make sure that the file in the second argument exists.")
    sys.exit(-3)

print("hello")
print("RUNNING: "+str(sys.argv[1])+" for "+str(sys.argv[2])+" times.")

for ind_all in range(0, repeat_times):
    # os.system("python3 dubins_naf.py")
    # print("RUNNING: dubins_naf.py")

    # os.system("python3 dubins_celtl.py")
    # print("RUNNING: dubins_celtl.py")

    os.system("python3 "+sys.argv[1])