#!/bin/bash

jobid=76909556
while : ; do
   qdel $jobid
   jobid=$(($jobid+1))
done