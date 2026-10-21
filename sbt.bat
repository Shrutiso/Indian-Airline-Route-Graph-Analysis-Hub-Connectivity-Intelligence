@echo off
set "JAVA_HOME=C:\jdk1.8.0_202"
set "HADOOP_HOME=C:\hadoop"
set "PATH=%JAVA_HOME%\bin;%HADOOP_HOME%\bin;%~dp0tools\sbt\bin;%PATH%"
"%~dp0tools\sbt\bin\sbt.bat" %*
