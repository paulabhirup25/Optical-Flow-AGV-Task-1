# 👁️ Optical Flow Navigation for AGV

## 🚀 Overview

This project implements **vision-based autonomous navigation** using **Lucas-Kanade Optical Flow**. The system enables an agent to navigate a racetrack in simulation using only motion information from camera frames.

---

## 🌱 Motivation

I chose this project to explore how robots can navigate using **pure vision**, without relying on expensive sensors. The goal was to understand how motion in images translates into actionable navigation decisions.

---

## ⚙️ Features

* Pyramidal Lucas-Kanade Optical Flow
* Feature tracking across frames
* Focus of Expansion (FOE) estimation
* Potential Field-based navigation
* Real-time integration with PyBullet

---

## 🧠 Methodology

1. Capture frames from simulation
2. Compute optical flow between frames
3. Estimate motion direction (FOE)
4. Generate potential field
5. Apply control for navigation

---

## 🚧 Challenges

* Understanding motion patterns in images
* Stabilizing navigation control
* Integrating perception with simulation

---

## 📈 Results

* Smooth navigation across racetrack
* Effective obstacle avoidance
* Robust performance using vision-only input

---

## 📚 Learnings

* Optical flow fundamentals
* Ego-motion estimation
* Real time perception + control systems

---

## 👤 Author

Abhirup Paul
