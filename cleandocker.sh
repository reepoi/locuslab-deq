#!/bin/bash

usage() {
    echo --tag [-t]: tag for docker image
    echo --build [-b]: if present, build the Dockerfile
    echo --dockerfile-path [-p]: the path to the Dockerfile
    echo --dockerfile-name [-f]: the custom name of the Dockerfile
}

if [ $# -eq 0 ]; then
    usage
    exit 1
fi

TAG=
DOCKERPATH=
DOCKERNAME=Dockerfile
BUILD=false
while [ "$1" != "" ]; do
    case $1 in
        -t | --tag)
            shift
            TAG=$1
            ;;
        -p | --dockerfile-path)
            shift
            DOCKERPATH=$1
            ;;
        -b | --build)
            BUILD=true
            ;;
        -f | --dockerfile-name)
            shift
            DOCKERNAME=$1
            ;;
        *)
            usage
            exit 1
            ;;
    esac
    shift
done

if [ "$TAG" == "" ]; then
    echo "error: tag must not be blank"
    exit 1
fi

if $BUILD; then
    if [ "$DOCKERPATH" == "" ]; then
        echo "error: Dockerfile path must not be blank"
        exit 1
    fi
    nvidia-docker build $DOCKERPATH -t $TAG -f $DOCKERNAME
fi

nvidia-docker rmi $(nvidia-docker images -q -f "label=taost" -f "dangling=true")

nvidia-docker run --interactive -p 127.0.0.1:8800:8800 --rm --name $TAG $TAG:latest

