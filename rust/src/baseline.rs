pub trait Baseline {
    fn dimension(&self) -> usize;

    fn intensity(&self, i: usize, t: f64) -> f64;

    fn upper_envelope(&self, i: usize, a: f64, b: f64) -> f64;

    fn cumint(&self, i: usize, t: f64) -> Option<f64>;

    fn inv_cumint(&self, i: usize, y: f64) -> Option<f64>;
}

pub struct BaselineConst {
    rates: Vec<f64>,
}

impl BaselineConst {
    pub fn new(rates: Vec<f64>) -> Result<Self, &'static str> {
        if rates.iter().any(|&r| r < 0.0) {
            return Err("baseline rates must be non-negative");
        }
        Ok(Self { rates })
    }
}

impl Baseline for BaselineConst {
    fn dimension(&self) -> usize {
        self.rates.len()
    }

    fn intensity(&self, i: usize, _t: f64) -> f64 {
        self.rates[i]
    }

    fn upper_envelope(&self, i: usize, _a: f64, _b: f64) -> f64 {
        self.rates[i]
    }

    fn cumint(&self, i: usize, t: f64) -> Option<f64> {
        Some(self.rates[i] * t.max(0.0))
    }

    fn inv_cumint(&self, i: usize, y: f64) -> Option<f64> {
        let rate = self.rates[i];
        if rate <= 0.0 {
            if y <= 0.0 {
                Some(0.0)
            } else {
                None
            }
        } else {
            Some(y / rate)
        }
    }
}

pub struct BaselinePiecewiseConst {
    breaks: Vec<f64>,
    rates: Vec<Vec<f64>>,
    cumulative: Vec<Vec<f64>>,
    totals: Vec<f64>,
}

impl BaselinePiecewiseConst {
    pub fn new(breaks: Vec<f64>, rates: Vec<Vec<f64>>) -> Result<Self, &'static str> {
        if breaks.len() < 2 {
            return Err("piecewise-constant baseline requires at least two breakpoints");
        }
        if (breaks[0]).abs() > 1e-12 {
            return Err("first breakpoint must be zero");
        }
        if !breaks.windows(2).all(|w| w[1] > w[0]) {
            return Err("baseline breakpoints must be strictly increasing");
        }
        if rates.iter().any(|row| row.len() + 1 != breaks.len()) {
            return Err("rate rows must have length breaks.len() - 1");
        }
        if rates.iter().flatten().any(|&r| r < 0.0) {
            return Err("baseline rates must be non-negative");
        }
        let dim = rates.len();
        let intervals: Vec<f64> = breaks.windows(2).map(|w| w[1] - w[0]).collect();
        if intervals.iter().any(|&len| len <= 0.0) {
            return Err("breakpoints must define positive-length intervals");
        }
        let mut cumulative = Vec::with_capacity(dim);
        let mut totals = Vec::with_capacity(dim);
        for row in &rates {
            let mut prefix = Vec::with_capacity(breaks.len());
            prefix.push(0.0);
            for (k, rate) in row.iter().enumerate() {
                let next = prefix[k] + rate * intervals[k];
                prefix.push(next);
            }
            totals.push(*prefix.last().unwrap());
            cumulative.push(prefix);
        }
        Ok(Self {
            breaks,
            rates,
            cumulative,
            totals,
        })
    }

    fn interval_index(&self, t: f64) -> Option<usize> {
        if t.is_nan() {
            return None;
        }
        if t < self.breaks[0] {
            return Some(0);
        }
        let last = self.breaks.len() - 1;
        if t >= self.breaks[last] {
            return Some(last - 1);
        }
        let idx = self.breaks.partition_point(|&b| b <= t);
        Some(idx.saturating_sub(1))
    }

    fn last_break(&self) -> f64 {
        self.breaks[self.breaks.len() - 1]
    }
}

impl Baseline for BaselinePiecewiseConst {
    fn dimension(&self) -> usize {
        self.rates.len()
    }

    fn intensity(&self, i: usize, t: f64) -> f64 {
        if let Some(idx) = self.interval_index(t) {
            self.rates[i][idx]
        } else {
            0.0
        }
    }

    fn upper_envelope(&self, i: usize, a: f64, b: f64) -> f64 {
        if b <= a {
            return 0.0;
        }
        let start = a.max(self.breaks[0]);
        let end = b.min(self.last_break());
        if end <= start {
            return 0.0;
        }
        let mut max_rate: f64 = 0.0;
        let mut idx = self.interval_index(start).unwrap_or(0);
        while idx < self.rates[i].len() {
            let interval_start = self.breaks[idx];
            let interval_end = self.breaks[idx + 1];
            if interval_start >= end {
                break;
            }
            if interval_end > start {
                max_rate = max_rate.max(self.rates[i][idx]);
            }
            idx += 1;
        }
        max_rate
    }

    fn cumint(&self, i: usize, t: f64) -> Option<f64> {
        if t.is_nan() {
            return None;
        }
        if t <= self.breaks[0] {
            return Some(0.0);
        }
        let last_break = self.last_break();
        if t >= last_break {
            return Some(self.totals[i]);
        }
        let idx = self.interval_index(t)?;
        let base = self.cumulative[i][idx];
        let dt = t - self.breaks[idx];
        Some(base + self.rates[i][idx] * dt)
    }

    fn inv_cumint(&self, i: usize, y: f64) -> Option<f64> {
        if y < 0.0 {
            return None;
        }
        let total = self.totals[i];
        if y > total {
            return None;
        }
        if y == total {
            return Some(self.last_break());
        }
        let prefix = &self.cumulative[i];
        let mut idx = prefix
            .iter()
            .position(|&val| val > y)
            .unwrap_or(prefix.len() - 1);
        if idx == 0 {
            return Some(0.0);
        }
        idx -= 1;
        let base = prefix[idx];
        let remaining = y - base;
        let rate = self.rates[i][idx];
        if rate <= 0.0 {
            return Some(self.breaks[idx]);
        }
        Some(self.breaks[idx] + remaining / rate)
    }
}

pub struct BaselinePiecewiseLinear {
    breaks: Vec<f64>,
    values: Vec<Vec<f64>>,
    slopes: Vec<Vec<f64>>,
    cumulative: Vec<Vec<f64>>,
    totals: Vec<f64>,
}

impl BaselinePiecewiseLinear {
    pub fn new(breaks: Vec<f64>, values: Vec<Vec<f64>>) -> Result<Self, &'static str> {
        if breaks.len() < 2 {
            return Err("piecewise-linear baseline requires at least two breakpoints");
        }
        if breaks[0].abs() > 1e-12 {
            return Err("first breakpoint must be zero");
        }
        if !breaks.windows(2).all(|w| w[1] > w[0]) {
            return Err("baseline breakpoints must be strictly increasing");
        }
        if values.iter().any(|row| row.len() != breaks.len()) {
            return Err("value rows must have length equal to breaks.len()");
        }
        if values.iter().flatten().any(|&v| v < 0.0) {
            return Err("baseline values must be non-negative");
        }
        let dim = values.len();
        let intervals: Vec<f64> = breaks.windows(2).map(|w| w[1] - w[0]).collect();
        if intervals.iter().any(|&len| len <= 0.0) {
            return Err("breakpoints must define positive-length intervals");
        }
        let mut slopes = Vec::with_capacity(dim);
        let mut cumulative = Vec::with_capacity(dim);
        let mut totals = Vec::with_capacity(dim);
        for row in &values {
            let mut row_slopes = Vec::with_capacity(intervals.len());
            for (k, delta) in intervals.iter().enumerate() {
                let slope = (row[k + 1] - row[k]) / delta;
                row_slopes.push(slope);
            }
            let mut prefix = Vec::with_capacity(breaks.len());
            prefix.push(0.0);
            for (k, delta) in intervals.iter().enumerate() {
                let start_value = row[k];
                let slope = row_slopes[k];
                let area = start_value * delta + 0.5 * slope * delta * delta;
                let next = prefix[k] + area;
                prefix.push(next);
            }
            totals.push(*prefix.last().unwrap());
            slopes.push(row_slopes);
            cumulative.push(prefix);
        }
        Ok(Self {
            breaks,
            values,
            slopes,
            cumulative,
            totals,
        })
    }

    fn dimension(&self) -> usize {
        self.values.len()
    }

    fn interval_index(&self, t: f64) -> Option<usize> {
        if t.is_nan() {
            return None;
        }
        if t <= self.breaks[0] {
            return Some(0);
        }
        let last = self.breaks.len() - 1;
        if t >= self.breaks[last] {
            return Some(last - 1);
        }
        let idx = self.breaks.partition_point(|&b| b <= t);
        Some(idx.saturating_sub(1))
    }

    fn last_break(&self) -> f64 {
        *self.breaks.last().unwrap()
    }

    fn intensity_at(&self, i: usize, idx: usize, t: f64) -> f64 {
        let start = self.breaks[idx];
        let clamped = t.clamp(start, self.breaks[idx + 1]);
        let dt = clamped - start;
        self.values[i][idx] + self.slopes[i][idx] * dt
    }
}

impl Baseline for BaselinePiecewiseLinear {
    fn dimension(&self) -> usize {
        self.dimension()
    }

    fn intensity(&self, i: usize, t: f64) -> f64 {
        if let Some(idx) = self.interval_index(t) {
            self.intensity_at(i, idx, t)
        } else {
            0.0
        }
    }

    fn upper_envelope(&self, i: usize, a: f64, b: f64) -> f64 {
        if b <= a {
            return 0.0;
        }
        let start = a.max(self.breaks[0]);
        let end = b.min(self.last_break());
        if end <= start {
            return 0.0;
        }
        let mut max_value: f64 = 0.0;
        let mut idx = self.interval_index(start).unwrap_or(0);
        while idx < self.slopes[i].len() {
            let interval_start = self.breaks[idx];
            let interval_end = self.breaks[idx + 1];
            if interval_start >= end {
                break;
            }
            let seg_start = start.max(interval_start);
            let seg_end = end.min(interval_end);
            if seg_end < seg_start {
                idx += 1;
                continue;
            }
            let value_start = self.intensity_at(i, idx, seg_start);
            let value_end = self.intensity_at(i, idx, seg_end);
            max_value = max_value.max(value_start.max(value_end));
            if seg_end >= end {
                break;
            }
            idx += 1;
        }
        max_value
    }

    fn cumint(&self, i: usize, t: f64) -> Option<f64> {
        if t.is_nan() {
            return None;
        }
        if t <= self.breaks[0] {
            return Some(0.0);
        }
        let last_break = self.last_break();
        if t >= last_break {
            return Some(self.totals[i]);
        }
        let idx = self.interval_index(t)?;
        let base = self.cumulative[i][idx];
        let start = self.breaks[idx];
        let dt = t - start;
        let slope = self.slopes[i][idx];
        let value_start = self.values[i][idx];
        Some(base + value_start * dt + 0.5 * slope * dt * dt)
    }

    fn inv_cumint(&self, i: usize, y: f64) -> Option<f64> {
        if y < 0.0 {
            return None;
        }
        let total = self.totals[i];
        if y > total {
            return None;
        }
        if y == total {
            return Some(self.last_break());
        }
        let prefix = &self.cumulative[i];
        let idx = prefix.partition_point(|&val| val <= y);
        if idx == 0 {
            return Some(self.breaks[0]);
        }
        let interval_idx = (idx - 1).min(self.slopes[i].len() - 1);
        let base = prefix[interval_idx];
        let next_prefix = self.cumulative[i][interval_idx + 1];
        let mut remaining = y - base;
        let interval_len = self.breaks[interval_idx + 1] - self.breaks[interval_idx];
        let max_interval_area = next_prefix - base;
        if remaining > max_interval_area {
            remaining = max_interval_area;
        }
        let slope = self.slopes[i][interval_idx];
        let start_value = self.values[i][interval_idx];
        let dt = if slope.abs() < 1e-12 {
            if start_value <= 0.0 {
                if remaining <= 0.0 {
                    0.0
                } else {
                    return None;
                }
            } else {
                remaining / start_value
            }
        } else {
            let discr = (start_value * start_value + 2.0 * slope * remaining).max(0.0);
            let root = discr.sqrt();
            (-start_value + root) / slope
        };
        let dt = dt.clamp(0.0, interval_len);
        Some(self.breaks[interval_idx] + dt)
    }
}

pub enum BaselineKind {
    Constant(BaselineConst),
    PiecewiseConst(BaselinePiecewiseConst),
    PiecewiseLinear(BaselinePiecewiseLinear),
}

impl Baseline for BaselineKind {
    fn dimension(&self) -> usize {
        match self {
            Self::Constant(b) => b.dimension(),
            Self::PiecewiseConst(b) => b.dimension(),
            Self::PiecewiseLinear(b) => b.dimension(),
        }
    }

    fn intensity(&self, i: usize, t: f64) -> f64 {
        match self {
            Self::Constant(b) => b.intensity(i, t),
            Self::PiecewiseConst(b) => b.intensity(i, t),
            Self::PiecewiseLinear(b) => b.intensity(i, t),
        }
    }

    fn upper_envelope(&self, i: usize, a: f64, b: f64) -> f64 {
        match self {
            Self::Constant(baseline) => baseline.upper_envelope(i, a, b),
            Self::PiecewiseConst(baseline) => baseline.upper_envelope(i, a, b),
            Self::PiecewiseLinear(baseline) => baseline.upper_envelope(i, a, b),
        }
    }

    fn cumint(&self, i: usize, t: f64) -> Option<f64> {
        match self {
            Self::Constant(b) => b.cumint(i, t),
            Self::PiecewiseConst(b) => b.cumint(i, t),
            Self::PiecewiseLinear(b) => b.cumint(i, t),
        }
    }

    fn inv_cumint(&self, i: usize, y: f64) -> Option<f64> {
        match self {
            Self::Constant(b) => b.inv_cumint(i, y),
            Self::PiecewiseConst(b) => b.inv_cumint(i, y),
            Self::PiecewiseLinear(b) => b.inv_cumint(i, y),
        }
    }
}
