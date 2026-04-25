// ── PARTICLES ──
function initParticles() {
  const c = document.getElementById('particles'), ctx = c.getContext('2d');
  let w, h, particles = [];
  function resize() { w = c.width = innerWidth; h = c.height = innerHeight; }
  resize(); addEventListener('resize', resize);
  for (let i = 0; i < 60; i++) {
    particles.push({ x: Math.random()*w, y: Math.random()*h, r: Math.random()*2+.5, dx: (Math.random()-.5)*.4, dy: (Math.random()-.5)*.4, o: Math.random()*.4+.1 });
  }
  function draw() {
    ctx.clearRect(0,0,w,h);
    particles.forEach(p => {
      ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, Math.PI*2);
      ctx.fillStyle = `rgba(99,102,241,${p.o})`; ctx.fill();
      p.x += p.dx; p.y += p.dy;
      if (p.x < 0 || p.x > w) p.dx *= -1;
      if (p.y < 0 || p.y > h) p.dy *= -1;
    });
    // Draw connections
    for (let i = 0; i < particles.length; i++) {
      for (let j = i+1; j < particles.length; j++) {
        const dx = particles[i].x - particles[j].x, dy = particles[i].y - particles[j].y;
        const dist = Math.sqrt(dx*dx + dy*dy);
        if (dist < 120) {
          ctx.beginPath(); ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.strokeStyle = `rgba(99,102,241,${.06*(1-dist/120)})`; ctx.stroke();
        }
      }
    }
    requestAnimationFrame(draw);
  }
  draw();
}

// ── ANIMATED COUNTERS ──
function animateCounters() {
  document.querySelectorAll('[data-count]').forEach(el => {
    const target = parseFloat(el.dataset.count);
    const decimals = (el.dataset.decimals || 0) | 0;
    const prefix = el.dataset.prefix || '';
    const suffix = el.dataset.suffix || '';
    const duration = 2000;
    let start = null;
    function step(ts) {
      if (!start) start = ts;
      const progress = Math.min((ts - start) / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 4);
      el.textContent = prefix + (target * eased).toFixed(decimals) + suffix;
      if (progress < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  });
}

// ── SCROLL REVEAL ──
function initReveal() {
  const obs = new IntersectionObserver((entries) => {
    entries.forEach(e => {
      if (e.isIntersecting) {
        e.target.classList.add('visible');
        // Animate progress bars inside
        e.target.querySelectorAll('.pbar-fill').forEach(bar => {
          bar.style.width = bar.dataset.w + '%';
        });
      }
    });
  }, { threshold: 0.15 });
  document.querySelectorAll('.reveal').forEach(el => obs.observe(el));
}

// ── NAVBAR ──
function initNavbar() {
  const nav = document.querySelector('.navbar');
  const links = document.querySelectorAll('.nav-links a');
  const sections = document.querySelectorAll('.section');

  window.addEventListener('scroll', () => {
    nav.classList.toggle('scrolled', scrollY > 50);
    // Highlight active nav link
    let current = '';
    sections.forEach(s => {
      if (scrollY >= s.offsetTop - 200) current = s.id;
    });
    links.forEach(a => a.classList.toggle('active', a.getAttribute('href') === '#' + current));
  });
}

// ── TABS ──
function initTabs() {
  document.querySelectorAll('.tabs').forEach(tabGroup => {
    const btns = tabGroup.querySelectorAll('.tab-btn');
    const container = tabGroup.parentElement;
    btns.forEach(btn => {
      btn.addEventListener('click', () => {
        btns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        container.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
        container.querySelector('#' + btn.dataset.tab).classList.add('active');
      });
    });
  });
}

// ── LIGHTBOX ──
let lbImages = [], lbIndex = 0;
function initLightbox() {
  const cards = document.querySelectorAll('.g-card');
  cards.forEach((card, i) => {
    lbImages.push(card.querySelector('img').src);
    card.addEventListener('click', () => openLB(i));
  });
  document.getElementById('lightbox').addEventListener('click', (e) => {
    if (e.target.id === 'lightbox') closeLB();
  });
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape') closeLB();
    if (e.key === 'ArrowRight') navLB(1);
    if (e.key === 'ArrowLeft') navLB(-1);
  });
}
function openLB(i) {
  lbIndex = i;
  document.getElementById('lb-img').src = lbImages[i];
  document.getElementById('lightbox').classList.add('open');
  document.body.style.overflow = 'hidden';
}
function closeLB() {
  document.getElementById('lightbox').classList.remove('open');
  document.body.style.overflow = '';
}
function navLB(dir) {
  lbIndex = (lbIndex + dir + lbImages.length) % lbImages.length;
  document.getElementById('lb-img').src = lbImages[lbIndex];
}

// ── CHART.JS CHARTS ──
function initCharts() {
  // R² comparison bar chart
  const ctx1 = document.getElementById('chartR2');
  if (!ctx1) return;
  new Chart(ctx1, {
    type: 'bar',
    data: {
      labels: ['Linear Regression', 'Random Forest', 'Stacking Ensemble', 'PRNN Base', 'Gated Pipeline'],
      datasets: [{
        label: 'R² Score',
        data: [0.99741, 0.99786, 0.99792, 0.99771, 0.99717],
        backgroundColor: ['rgba(99,102,241,.5)', 'rgba(6,182,212,.5)', 'rgba(16,185,129,.5)', 'rgba(139,92,246,.5)', 'rgba(245,158,11,.5)'],
        borderColor: ['#6366f1', '#06b6d4', '#10b981', '#8b5cf6', '#f59e0b'],
        borderWidth: 2, borderRadius: 8, barPercentage: .6
      }]
    },
    options: {
      responsive: true,
      animation: { duration: 2000, easing: 'easeOutQuart' },
      plugins: {
        legend: { display: false },
        tooltip: { backgroundColor: '#1e293b', titleColor: '#e2e8f0', bodyColor: '#94a3b8', borderColor: 'rgba(255,255,255,.1)', borderWidth: 1, cornerRadius: 10, padding: 12 }
      },
      scales: {
        y: { min: 0.996, max: 0.999, ticks: { color: '#64748b', callback: v => v.toFixed(4) }, grid: { color: 'rgba(255,255,255,.04)' } },
        x: { ticks: { color: '#64748b', font: { size: 11 } }, grid: { display: false } }
      }
    }
  });

  // Error metrics radar
  const ctx2 = document.getElementById('chartRadar');
  if (!ctx2) return;
  new Chart(ctx2, {
    type: 'radar',
    data: {
      labels: ['R² (×100)', 'MAE Inv', 'RMSE Inv', 'MAPE Inv', 'RAE Inv'],
      datasets: [
        { label: 'Stacking', data: [99.792, 98.19, 97.33, 99.27, 96.11], borderColor: '#10b981', backgroundColor: 'rgba(16,185,129,.1)', pointBackgroundColor: '#10b981', borderWidth: 2 },
        { label: 'Random Forest', data: [99.786, 98.20, 97.29, 99.27, 96.14], borderColor: '#06b6d4', backgroundColor: 'rgba(6,182,212,.1)', pointBackgroundColor: '#06b6d4', borderWidth: 2 },
        { label: 'Linear Reg', data: [99.741, 97.83, 97.02, 99.13, 95.34], borderColor: '#6366f1', backgroundColor: 'rgba(99,102,241,.1)', pointBackgroundColor: '#6366f1', borderWidth: 2 }
      ]
    },
    options: {
      responsive: true,
      animation: { duration: 2000 },
      plugins: { legend: { labels: { color: '#94a3b8', font: { size: 12 } } } },
      scales: { r: { min: 94, max: 100, ticks: { color: '#475569', backdropColor: 'transparent', stepSize: 1 }, grid: { color: 'rgba(255,255,255,.06)' }, pointLabels: { color: '#94a3b8', font: { size: 11 } } } }
    }
  });

  // OOD Severity Distribution (doughnut)
  const ctx3 = document.getElementById('chartOOD');
  if (!ctx3) return;
  new Chart(ctx3, {
    type: 'doughnut',
    data: {
      labels: ['High Confidence', 'Moderate', 'Physics Anchored', 'OOD Baseline'],
      datasets: [{ data: [972, 372, 115, 18], backgroundColor: ['#10b981', '#f59e0b', '#f97316', '#ef4444'], borderColor: '#0d1117', borderWidth: 3, hoverOffset: 8 }]
    },
    options: {
      responsive: true, cutout: '60%',
      animation: { animateRotate: true, duration: 2000 },
      plugins: { legend: { position: 'bottom', labels: { color: '#94a3b8', padding: 16, font: { size: 12 } } } }
    }
  });
}

// ── INIT ──
document.addEventListener('DOMContentLoaded', () => {
  initParticles();
  initReveal();
  initNavbar();
  initTabs();
  initLightbox();
  // Delay counter animation until hero is visible
  const heroObs = new IntersectionObserver(entries => {
    if (entries[0].isIntersecting) { animateCounters(); heroObs.disconnect(); }
  }, { threshold: 0.3 });
  heroObs.observe(document.getElementById('hero'));
  // Init charts when section visible
  const chartObs = new IntersectionObserver(entries => {
    if (entries[0].isIntersecting) { initCharts(); chartObs.disconnect(); }
  }, { threshold: 0.2 });
  const chartSec = document.getElementById('charts-section');
  if (chartSec) chartObs.observe(chartSec);
});
