use crate::baseline::Baseline;
use crate::kernel::Kernel;
use crate::pp_data::MultivariatePPData;
use rand_distr::Distribution;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImmigrantSamplingMethod {
    Thinning,
    InverseTransform,
}

fn thinning_component<B: Baseline, R: rand::Rng + ?Sized>(
    baseline: &B,
    idx: usize,
    t_max: f64,
    rng: &mut R,
    exp_unit: &rand_distr::Exp<f64>,
) -> Result<Vec<f64>, &'static str> {
    let mut events = Vec::new();
    let mut t = 0.0;
    let mut envelope = baseline.upper_envelope(idx, t, t_max);
    if envelope < 0.0 {
        return Err("baseline upper envelope must be non-negative");
    }
    if envelope == 0.0 {
        return Ok(events);
    }
    while t < t_max {
        let wait = exp_unit.sample(rng) / envelope;
        t += wait;
        if t >= t_max {
            break;
        }
        let lambda_t = baseline.intensity(idx, t);
        if lambda_t < 0.0 {
            return Err("baseline intensity must be non-negative");
        }
        let u: f64 = rand::Rng::random(rng);
        if u * envelope <= lambda_t {
            events.push(t);
        }
        envelope = baseline.upper_envelope(idx, t, t_max);
    }
    Ok(events)
}

fn sample_immigrants_thinning<B: Baseline, R: rand::Rng + ?Sized>(
    baseline: &B,
    t_max: f64,
    rng: &mut R,
) -> Result<Vec<Vec<f64>>, &'static str> {
    let d = baseline.dimension();
    let mut events: Vec<Vec<f64>> = vec![Vec::new(); d];
    let exp_unit = rand_distr::Exp::new(1.0).map_err(|_| "invalid exponential rate")?;

    for i in 0..d {
        events[i] = thinning_component(baseline, i, t_max, rng, &exp_unit)?;
    }
    Ok(events)
}

fn sample_immigrants_inverse<B: Baseline, R: rand::Rng + ?Sized>(
    baseline: &B,
    t_max: f64,
    rng: &mut R,
) -> Result<Vec<Vec<f64>>, &'static str> {
    let d = baseline.dimension();
    let mut events: Vec<Vec<f64>> = vec![Vec::new(); d];
    let exp_unit = rand_distr::Exp::new(1.0).map_err(|_| "invalid exponential rate")?;

    for i in 0..d {
        let total = baseline
            .cumint(i, t_max)
            .ok_or("baseline cumulative intensity is unavailable for inverse transform sampling")?;
        if total < 0.0 {
            return Err("baseline cumulative intensity must be non-negative");
        }
        if total == 0.0 {
            continue;
        }
        if baseline.inv_cumint(i, total * 0.5).is_none() {
            return Err(
                "baseline inverse cumulative intensity is unavailable for inverse transform sampling",
            );
        }
        let mut sum = 0.0;
        while sum < total {
            sum += exp_unit.sample(rng);
            if sum >= total {
                break;
            }
            let time = baseline
                .inv_cumint(i, sum)
                .ok_or(
                    "baseline inverse cumulative intensity is unavailable for inverse transform sampling",
                )?;
            if time < t_max {
                events[i].push(time);
            }
        }
    }
    Ok(events)
}

pub fn simulate_hawkes_branching_with_baseline<R: rand::Rng + ?Sized, K: Kernel, B: Baseline>(
    t_max: f64,
    baseline: &B,
    alpha: &[Vec<f64>],
    kernel: &K,
    method: ImmigrantSamplingMethod,
    rng: &mut R,
) -> Result<MultivariatePPData, &'static str> {
    let d = baseline.dimension();
    if d == 0 {
        return Err("baseline must have positive dimension");
    }
    if alpha.len() != d || !alpha.iter().all(|row| row.len() == d) {
        return Err("alpha must be a square matrix matching the baseline dimension");
    }
    kernel.validate_dimension(d)?;
    if t_max <= 0.0 {
        return Err("t_max must be positive");
    }
    // Initialize the events vector with baseline immigrants
    let mut events = match method {
        ImmigrantSamplingMethod::Thinning => sample_immigrants_thinning(baseline, t_max, rng)?,
        ImmigrantSamplingMethod::InverseTransform => {
            sample_immigrants_inverse(baseline, t_max, rng)?
        }
    };

    // placeholder for anscestor events
    // we will use this queue to manage the branching process
    // each element is a tuple of (parent_type, parent_time)
    let mut queue: std::collections::VecDeque<(usize, f64)> = events
        .iter()
        .enumerate()
        .flat_map(|(i, ts)| ts.iter().map(move |&t| (i, t)))
        .collect();

    // generate branching events
    while let Some((parent_type, parent_time)) = queue.pop_front() {
        for child_type in 0..d {
            // num of children ~ Poisson(alpha[parent][child])
            let lambda = alpha[parent_type][child_type];
            if lambda == 0.0 {
                continue;
            }
            let n_child = rand_distr::Poisson::new(lambda).unwrap().sample(rng) as usize;

            for _ in 0..n_child {
                let delay = kernel.sample_delay(parent_type, child_type, rng);
                let child_time = parent_time + delay;
                if child_time < t_max {
                    events[child_type].push(child_time);
                    // enqueue the child event for further branching
                    queue.push_back((child_type, child_time));
                }
            }
        }
    }
    // sort each type's events
    for ts in &mut events {
        ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    }

    Ok(MultivariatePPData {
        timestamps: events,
        obs_window: (0.0, t_max),
    })
}
