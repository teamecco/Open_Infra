#include <stdio.h>
#include <wiringPi.h>
#include <softPwm.h>
#include <unistd.h>


#define SERVO1 1
#define SERVO2 4
#define SERVO3 5

int main(){
	char str;
	
	if(wiringPiSetup()==-1)
		return 1;
	
	softPwmCreate(SERVO1, 0, 200);
	softPwmCreate(SERVO2, 0, 200);
	softPwmCreate(SERVO3, 0, 200);


	while(1)
	{
		fputs("select (a,b,c) (d,e,f) (g,h,i) : ", stdout);
		scanf("%c", &str);
		getchar();
		switch(str){
			case 'a':softPwmWrite(SERVO1, 15); break;
			case 'b':softPwmWrite(SERVO1, 24); break;
			case 'c':softPwmWrite(SERVO1, 5); break;
			case 'd':softPwmWrite(SERVO2, 15); break;
			case 'e':softPwmWrite(SERVO2, 24); break;
			case 'f':softPwmWrite(SERVO2, 5); break;
			case 'g':softPwmWrite(SERVO3, 15); break;
			case 'h':softPwmWrite(SERVO3, 24); break;
			case 'i':softPwmWrite(SERVO3, 5); break;
			case 'q': return 0;
		}
	}
	return 0;
}
