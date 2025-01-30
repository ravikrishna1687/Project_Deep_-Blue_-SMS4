import sys
def find_fridge_temp(n,intervals):
    if n==0:
        print("-100 -99")
        return
    min_a,max_b =  101,101
    min_c,max_d = -101,101
    for a , b in intervals:
        if b < min_c:
            min_a=max(min_a,a)
            max_b=min(max_b,b)
        elif a > max_b:
            min_c=max(min_c,a)
            max_d = min(max_d,b)
        else:
            print(-1)
            return
    print(min_a,min_c)
    
input=sys.stdin.read
data=input().split()
n=int(data[0])
intervals=[]
for i in range(n):
    a = int(data[2*i+1])
    b=int(data[2*i+2])
    intervals.append((a,b))
find_fridge_temp(n,intervals)