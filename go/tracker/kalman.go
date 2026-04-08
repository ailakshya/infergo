package tracker

import "math"

// kalmanState holds the 8-dimensional state for a single tracked object.
// State vector: [cx, cy, aspect_ratio, height, vx, vy, va, vh]
// This implements a constant-velocity Kalman filter matching the
// ultralytics/ByteTrack KalmanFilterXYAH.
type kalmanState struct {
	mean       [8]float64  // state estimate
	covariance [8][8]float64 // error covariance
}

// Noise weights matching ultralytics defaults.
const (
	stdWeightPosition = 1.0 / 20.0
	stdWeightVelocity = 1.0 / 160.0
)

// initKalman creates a new Kalman state from a measurement [cx,cy,a,h].
func initKalman(measurement [4]float64) kalmanState {
	var ks kalmanState
	ks.mean[0] = measurement[0]
	ks.mean[1] = measurement[1]
	ks.mean[2] = measurement[2]
	ks.mean[3] = measurement[3]
	// velocities start at zero

	// Initial covariance: large uncertainty
	h := measurement[3]
	std := [8]float64{
		2 * stdWeightPosition * h,
		2 * stdWeightPosition * h,
		1e-2,
		2 * stdWeightPosition * h,
		10 * stdWeightVelocity * h,
		10 * stdWeightVelocity * h,
		1e-5,
		10 * stdWeightVelocity * h,
	}
	for i := 0; i < 8; i++ {
		ks.covariance[i][i] = std[i] * std[i]
	}
	return ks
}

// predict projects the state forward one time step using constant velocity.
func (ks *kalmanState) predict() {
	h := ks.mean[3]
	if h < 1 {
		h = 1
	}

	// Process noise Q
	stdPos := [4]float64{
		stdWeightPosition * h,
		stdWeightPosition * h,
		1e-2,
		stdWeightPosition * h,
	}
	stdVel := [4]float64{
		stdWeightVelocity * h,
		stdWeightVelocity * h,
		1e-5,
		stdWeightVelocity * h,
	}

	// x = F * x (add velocity to position)
	for i := 0; i < 4; i++ {
		ks.mean[i] += ks.mean[i+4]
	}

	// P = F * P * F^T + Q
	// F is identity with ones in the upper-right 4x4 block.
	// Rather than full matrix multiply, we exploit the sparse structure:
	// P_new[i][j] = P[i][j] + P[i][j+4] + P[i+4][j] + P[i+4][j+4] for i,j < 4
	// P_new[i][j+4] = P[i][j+4] + P[i+4][j+4] for i < 4, j < 4
	// P_new[i+4][j] = P[i+4][j] + P[i+4][j+4] for i < 4, j < 4
	// P_new[i+4][j+4] = P[i+4][j+4]

	var newP [8][8]float64

	// Top-left 4x4
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			newP[i][j] = ks.covariance[i][j] + ks.covariance[i][j+4] +
				ks.covariance[i+4][j] + ks.covariance[i+4][j+4]
		}
	}
	// Top-right 4x4
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			newP[i][j+4] = ks.covariance[i][j+4] + ks.covariance[i+4][j+4]
		}
	}
	// Bottom-left 4x4
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			newP[i+4][j] = ks.covariance[i+4][j] + ks.covariance[i+4][j+4]
		}
	}
	// Bottom-right 4x4
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			newP[i+4][j+4] = ks.covariance[i+4][j+4]
		}
	}

	// Add process noise Q (diagonal)
	for i := 0; i < 4; i++ {
		newP[i][i] += stdPos[i] * stdPos[i]
		newP[i+4][i+4] += stdVel[i] * stdVel[i]
	}

	ks.covariance = newP
}

// update incorporates a new measurement [cx,cy,a,h] into the state.
func (ks *kalmanState) update(measurement [4]float64) {
	h := ks.mean[3]
	if h < 1 {
		h = 1
	}

	// Measurement noise R
	std := [4]float64{
		stdWeightPosition * h,
		stdWeightPosition * h,
		1e-2,
		stdWeightPosition * h,
	}

	// H is [I_4 | 0_4], so projected_mean = mean[:4], projected_cov = P[:4][:4]
	// Innovation: y = z - H*x
	var innovation [4]float64
	for i := 0; i < 4; i++ {
		innovation[i] = measurement[i] - ks.mean[i]
	}

	// S = H*P*H^T + R = P[:4][:4] + R
	var S [4][4]float64
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			S[i][j] = ks.covariance[i][j]
		}
		S[i][i] += std[i] * std[i]
	}

	// S_inv = inv(S) via 4x4 inversion
	Sinv := inv4x4(S)

	// K = P * H^T * S^-1 = P[:8][:4] * S^-1
	var K [8][4]float64
	for i := 0; i < 8; i++ {
		for j := 0; j < 4; j++ {
			for k := 0; k < 4; k++ {
				K[i][j] += ks.covariance[i][k] * Sinv[k][j]
			}
		}
	}

	// x = x + K * y
	for i := 0; i < 8; i++ {
		for j := 0; j < 4; j++ {
			ks.mean[i] += K[i][j] * innovation[j]
		}
	}

	// P = (I - K*H) * P
	// KH is 8x8 where KH[i][j] = K[i][j] for j < 4, else 0
	var newP [8][8]float64
	for i := 0; i < 8; i++ {
		for j := 0; j < 8; j++ {
			var kh float64
			for k := 0; k < 4; k++ {
				// H[k][j] = 1 if k==j && j<4, else 0
				if j < 4 && k == j {
					kh += K[i][k]
				}
			}
			ikh := -kh
			if i == j {
				ikh += 1
			}
			for l := 0; l < 8; l++ {
				newP[i][j] += ikh * ks.covariance[l][j]
			}
		}
	}

	// Simplified: P = P - K * S * K^T ensures symmetry
	// But the (I-KH)*P form is standard. Let's fix:
	// P_new[i][j] = sum_l (I-KH)[i][l] * P[l][j]
	// (I-KH)[i][l] = delta(i,l) - K[i][l] for l<4, delta(i,l) for l>=4
	for i := 0; i < 8; i++ {
		for j := 0; j < 8; j++ {
			newP[i][j] = ks.covariance[i][j]
			for k := 0; k < 4; k++ {
				newP[i][j] -= K[i][k] * ks.covariance[k][j]
			}
		}
	}

	ks.covariance = newP
}

// mahalanobisDistance computes the squared Mahalanobis distance between the
// current state projection and a measurement. Used for gating in association.
func (ks *kalmanState) mahalanobisDistance(measurement [4]float64) float64 {
	h := ks.mean[3]
	if h < 1 {
		h = 1
	}
	std := [4]float64{
		stdWeightPosition * h,
		stdWeightPosition * h,
		1e-2,
		stdWeightPosition * h,
	}
	var S [4][4]float64
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			S[i][j] = ks.covariance[i][j]
		}
		S[i][i] += std[i] * std[i]
	}
	Sinv := inv4x4(S)

	var d [4]float64
	for i := 0; i < 4; i++ {
		d[i] = measurement[i] - ks.mean[i]
	}

	var dist float64
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			dist += d[i] * Sinv[i][j] * d[j]
		}
	}
	return dist
}

// inv4x4 computes the inverse of a 4x4 matrix using cofactor expansion.
func inv4x4(m [4][4]float64) [4][4]float64 {
	// Compute cofactors and determinant
	var inv [4][4]float64

	inv[0][0] = m[1][1]*(m[2][2]*m[3][3]-m[2][3]*m[3][2]) - m[1][2]*(m[2][1]*m[3][3]-m[2][3]*m[3][1]) + m[1][3]*(m[2][1]*m[3][2]-m[2][2]*m[3][1])
	inv[0][1] = -(m[0][1]*(m[2][2]*m[3][3]-m[2][3]*m[3][2]) - m[0][2]*(m[2][1]*m[3][3]-m[2][3]*m[3][1]) + m[0][3]*(m[2][1]*m[3][2]-m[2][2]*m[3][1]))
	inv[0][2] = m[0][1]*(m[1][2]*m[3][3]-m[1][3]*m[3][2]) - m[0][2]*(m[1][1]*m[3][3]-m[1][3]*m[3][1]) + m[0][3]*(m[1][1]*m[3][2]-m[1][2]*m[3][1])
	inv[0][3] = -(m[0][1]*(m[1][2]*m[2][3]-m[1][3]*m[2][2]) - m[0][2]*(m[1][1]*m[2][3]-m[1][3]*m[2][1]) + m[0][3]*(m[1][1]*m[2][2]-m[1][2]*m[2][1]))

	inv[1][0] = -(m[1][0]*(m[2][2]*m[3][3]-m[2][3]*m[3][2]) - m[1][2]*(m[2][0]*m[3][3]-m[2][3]*m[3][0]) + m[1][3]*(m[2][0]*m[3][2]-m[2][2]*m[3][0]))
	inv[1][1] = m[0][0]*(m[2][2]*m[3][3]-m[2][3]*m[3][2]) - m[0][2]*(m[2][0]*m[3][3]-m[2][3]*m[3][0]) + m[0][3]*(m[2][0]*m[3][2]-m[2][2]*m[3][0])
	inv[1][2] = -(m[0][0]*(m[1][2]*m[3][3]-m[1][3]*m[3][2]) - m[0][2]*(m[1][0]*m[3][3]-m[1][3]*m[3][0]) + m[0][3]*(m[1][0]*m[3][2]-m[1][2]*m[3][0]))
	inv[1][3] = m[0][0]*(m[1][2]*m[2][3]-m[1][3]*m[2][2]) - m[0][2]*(m[1][0]*m[2][3]-m[1][3]*m[2][0]) + m[0][3]*(m[1][0]*m[2][2]-m[1][2]*m[2][0])

	inv[2][0] = m[1][0]*(m[2][1]*m[3][3]-m[2][3]*m[3][1]) - m[1][1]*(m[2][0]*m[3][3]-m[2][3]*m[3][0]) + m[1][3]*(m[2][0]*m[3][1]-m[2][1]*m[3][0])
	inv[2][1] = -(m[0][0]*(m[2][1]*m[3][3]-m[2][3]*m[3][1]) - m[0][1]*(m[2][0]*m[3][3]-m[2][3]*m[3][0]) + m[0][3]*(m[2][0]*m[3][1]-m[2][1]*m[3][0]))
	inv[2][2] = m[0][0]*(m[1][1]*m[3][3]-m[1][3]*m[3][1]) - m[0][1]*(m[1][0]*m[3][3]-m[1][3]*m[3][0]) + m[0][3]*(m[1][0]*m[3][1]-m[1][1]*m[3][0])
	inv[2][3] = -(m[0][0]*(m[1][1]*m[2][3]-m[1][3]*m[2][1]) - m[0][1]*(m[1][0]*m[2][3]-m[1][3]*m[2][0]) + m[0][3]*(m[1][0]*m[2][1]-m[1][1]*m[2][0]))

	inv[3][0] = -(m[1][0]*(m[2][1]*m[3][2]-m[2][2]*m[3][1]) - m[1][1]*(m[2][0]*m[3][2]-m[2][2]*m[3][0]) + m[1][2]*(m[2][0]*m[3][1]-m[2][1]*m[3][0]))
	inv[3][1] = m[0][0]*(m[2][1]*m[3][2]-m[2][2]*m[3][1]) - m[0][1]*(m[2][0]*m[3][2]-m[2][2]*m[3][0]) + m[0][2]*(m[2][0]*m[3][1]-m[2][1]*m[3][0])
	inv[3][2] = -(m[0][0]*(m[1][1]*m[3][2]-m[1][2]*m[3][1]) - m[0][1]*(m[1][0]*m[3][2]-m[1][2]*m[3][0]) + m[0][2]*(m[1][0]*m[3][1]-m[1][1]*m[3][0]))
	inv[3][3] = m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1]) - m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0]) + m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0])

	det := m[0][0]*inv[0][0] + m[0][1]*inv[1][0] + m[0][2]*inv[2][0] + m[0][3]*inv[3][0]
	if math.Abs(det) < 1e-12 {
		// Singular matrix — return large diagonal (graceful degradation)
		for i := 0; i < 4; i++ {
			for j := 0; j < 4; j++ {
				inv[i][j] = 0
			}
			inv[i][i] = 1e12
		}
		return inv
	}

	invDet := 1.0 / det
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			inv[i][j] *= invDet
		}
	}
	return inv
}
