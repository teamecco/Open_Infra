Logstash Cron Job
=================

container config
----------------

> ### **Container Name**
>> #### logstash-csv-cronjob

> ### **Container Image**
>> #### rndlr96/logstash-csv:v1
>>> ##### Customized docker image with elasticsearch input plugin and csv output plugin installed

> ###**Vloume Mount**
>> ####Volumn name : Config-data
>> ####Volumn path : /csv

> ###**User-defined variables**
>> ####logstash-csv-file : (KEY)logstash.conf | (mount path)/usr/share/logstash/pipeline/logstash.conf
>> ####logstash-csv      : (key)logstash.yml  | (mount path)/usr/share/logstash/config/logstash.yml
>>>#####Cron Job does not allow duplicate user-defined variable names



cronjob configuration
---------------------

> ###**Schedule**
>> ####0 0 ? * 1/4
>>> #####//job will running every Thursday

> ###**Concurrency Policy**
>> ####The job must be run at least once


