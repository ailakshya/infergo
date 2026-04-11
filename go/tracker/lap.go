package tracker

import "math"

const costInf = 1e18

// linearAssignment solves the linear assignment problem (LAP) using the
// Jonker-Volgenant algorithm (simplified Hungarian method).
// costMatrix is [nRows][nCols], thresh is the maximum allowed cost.
// Returns (matchedRows, matchedCols, unmatchedRows, unmatchedCols).
// Matched pairs: matchedRows[i] is paired with matchedCols[i].
func linearAssignment(costMatrix [][]float64, thresh float64) (
	matchedRows, matchedCols []int,
	unmatchedRows, unmatchedCols []int,
) {
	nRows := len(costMatrix)
	if nRows == 0 {
		// No rows: all columns are unmatched. Caller must tell us nCols.
		// Since we only have the matrix, we can't know nCols — return nil.
		// The caller (bytetrack.go) handles empty pool separately.
		return nil, nil, nil, nil
	}
	nCols := len(costMatrix[0])
	if nCols == 0 {
		for i := 0; i < nRows; i++ {
			unmatchedRows = append(unmatchedRows, i)
		}
		return nil, nil, unmatchedRows, nil
	}

	// Use Hungarian algorithm for optimal assignment
	rowAssign, colAssign := hungarian(costMatrix, nRows, nCols)

	matchedR := make(map[int]bool)
	matchedC := make(map[int]bool)

	for r, c := range rowAssign {
		if c >= 0 && c < nCols && costMatrix[r][c] <= thresh {
			matchedRows = append(matchedRows, r)
			matchedCols = append(matchedCols, c)
			matchedR[r] = true
			matchedC[c] = true
		}
	}
	_ = colAssign

	for i := 0; i < nRows; i++ {
		if !matchedR[i] {
			unmatchedRows = append(unmatchedRows, i)
		}
	}
	for j := 0; j < nCols; j++ {
		if !matchedC[j] {
			unmatchedCols = append(unmatchedCols, j)
		}
	}
	return
}

// hungarian implements the Hungarian algorithm for the assignment problem.
// Handles non-square matrices by padding. Returns row→col and col→row assignments.
func hungarian(cost [][]float64, nRows, nCols int) ([]int, []int) {
	n := nRows
	if nCols > n {
		n = nCols
	}

	// Pad to square matrix
	c := make([][]float64, n)
	for i := range c {
		c[i] = make([]float64, n)
		for j := range c[i] {
			if i < nRows && j < nCols {
				c[i][j] = cost[i][j]
			} else {
				c[i][j] = costInf
			}
		}
	}

	// u[i] = potential for row i, v[j] = potential for column j
	u := make([]float64, n+1)
	v := make([]float64, n+1)
	// p[j] = row assigned to column j
	p := make([]int, n+1)
	// way[j] = previous column in alternating path
	way := make([]int, n+1)

	for i := 1; i <= n; i++ {
		p[0] = i
		j0 := 0
		minv := make([]float64, n+1)
		used := make([]bool, n+1)
		for j := 1; j <= n; j++ {
			minv[j] = math.Inf(1)
		}

		for {
			used[j0] = true
			i0 := p[j0]
			delta := math.Inf(1)
			j1 := 0

			for j := 1; j <= n; j++ {
				if used[j] {
					continue
				}
				cur := c[i0-1][j-1] - u[i0] - v[j]
				if cur < minv[j] {
					minv[j] = cur
					way[j] = j0
				}
				if minv[j] < delta {
					delta = minv[j]
					j1 = j
				}
			}

			for j := 0; j <= n; j++ {
				if used[j] {
					u[p[j]] += delta
					v[j] -= delta
				} else {
					minv[j] -= delta
				}
			}

			j0 = j1
			if p[j0] == 0 {
				break
			}
		}

		for {
			prev := way[j0]
			p[j0] = p[prev]
			j0 = prev
			if j0 == 0 {
				break
			}
		}
	}

	rowAssign := make([]int, nRows)
	colAssign := make([]int, nCols)
	for i := range rowAssign {
		rowAssign[i] = -1
	}
	for j := range colAssign {
		colAssign[j] = -1
	}

	for j := 1; j <= n; j++ {
		row := p[j] - 1
		col := j - 1
		if row < nRows && col < nCols {
			rowAssign[row] = col
			colAssign[col] = row
		}
	}

	return rowAssign, colAssign
}

// iouDistance computes the IoU distance matrix between tracks and detections.
// Returns a [len(tracks)][len(dets)] matrix where dist = 1 - IoU.
func iouDistance(tracks []*strack, dets []Detection) [][]float64 {
	m := len(tracks)
	n := len(dets)
	dist := make([][]float64, m)
	for i := 0; i < m; i++ {
		dist[i] = make([]float64, n)
		tbox := tracks[i].box()
		for j := 0; j < n; j++ {
			dist[i][j] = 1 - iou(tbox, dets[j].Box)
		}
	}
	return dist
}

// fuseScore combines IoU distance with detection confidence.
// Lower score = better match (like ultralytics fuse_score).
func fuseScore(dist [][]float64, dets []Detection) [][]float64 {
	for i := range dist {
		for j := range dist[i] {
			dist[i][j] = 1 - ((1 - dist[i][j]) * dets[j].Confidence)
		}
	}
	return dist
}

// iou computes Intersection over Union between two boxes [x1,y1,x2,y2].
func iou(a, b [4]float64) float64 {
	x1 := math.Max(a[0], b[0])
	y1 := math.Max(a[1], b[1])
	x2 := math.Min(a[2], b[2])
	y2 := math.Min(a[3], b[3])

	inter := math.Max(0, x2-x1) * math.Max(0, y2-y1)
	if inter == 0 {
		return 0
	}

	areaA := (a[2] - a[0]) * (a[3] - a[1])
	areaB := (b[2] - b[0]) * (b[3] - b[1])
	union := areaA + areaB - inter
	if union <= 0 {
		return 0
	}
	return inter / union
}
