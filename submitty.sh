#!/bin/sh

mkdir -p Submitty
rm Submitty/*.c
rm Submitty/*.cu
rm Submitty/*.h

cp Source/* Submitty
cp Source/**/* Submitty
cp README.md Submitty/README.md

