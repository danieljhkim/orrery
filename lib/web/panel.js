// Declarative control panel — kills the slider/button boilerplate every sim
// used to re-implement by hand.
//
//   const panel = createPanel([
//     { type: 'button', label: 'Pause', onClick: (btn) => { ... } },
//     { type: 'range',  key: 'speed', label: 'Speed', min: 1, max: 200, value: 30,
//       format: (v) => `${v} days/sec` },
//     { type: 'toggle', key: 'trails', label: 'Trails', value: true },
//     { type: 'select', key: 'view', label: 'View', options: ['inner', 'full'] },
//     { type: 'readout', key: 'energy' },
//   ], { mount: document.getElementById('controls') });
//
//   panel.values.speed          // current slider value (number)
//   panel.set('speed', 60)      // programmatic update (fires onChange)
//   panel.setReadout('energy', 'E = -1.234')
//
// Styling comes from lib/web/style.css (.orrery-controls).
export function createPanel(defs, { mount = document.body } = {}) {
  const el = document.createElement('div');
  el.className = 'orrery-controls';
  const values = {};
  const controls = {}; // key -> { input?, out?, def }

  for (const def of defs) {
    switch (def.type) {
      case 'button': {
        const btn = document.createElement('button');
        btn.textContent = def.label;
        btn.addEventListener('click', () => def.onClick?.(btn));
        el.appendChild(btn);
        if (def.key) controls[def.key] = { input: btn, def };
        break;
      }
      case 'range': {
        const label = document.createElement('label');
        label.textContent = def.label ?? def.key;
        const input = document.createElement('input');
        input.type = 'range';
        input.min = def.min ?? 0;
        input.max = def.max ?? 1;
        input.step = def.step ?? 'any';
        input.value = def.value ?? def.min ?? 0;
        const out = document.createElement('span');
        out.className = 'readout';
        const fmt = def.format ?? ((v) => String(v));
        const update = () => {
          values[def.key] = Number(input.value);
          out.textContent = fmt(values[def.key]);
        };
        input.addEventListener('input', () => { update(); def.onChange?.(values[def.key]); });
        update();
        el.append(label, input, out);
        controls[def.key] = { input, out, def };
        break;
      }
      case 'toggle': {
        const btn = document.createElement('button');
        values[def.key] = !!def.value;
        const paint = () => {
          btn.textContent = def.label ?? def.key;
          btn.classList.toggle('active', values[def.key]);
        };
        btn.addEventListener('click', () => {
          values[def.key] = !values[def.key];
          paint();
          def.onChange?.(values[def.key]);
        });
        paint();
        el.appendChild(btn);
        controls[def.key] = { input: btn, def };
        break;
      }
      case 'select': {
        const label = document.createElement('label');
        label.textContent = def.label ?? def.key;
        const sel = document.createElement('select');
        for (const opt of def.options) {
          const o = document.createElement('option');
          if (typeof opt === 'object') { o.value = opt.value; o.textContent = opt.label; }
          else { o.value = opt; o.textContent = opt; }
          sel.appendChild(o);
        }
        if (def.value != null) sel.value = def.value;
        values[def.key] = sel.value;
        sel.addEventListener('change', () => {
          values[def.key] = sel.value;
          def.onChange?.(sel.value);
        });
        el.append(label, sel);
        controls[def.key] = { input: sel, def };
        break;
      }
      case 'readout': {
        const out = document.createElement('span');
        out.className = 'readout';
        out.textContent = def.value ?? '';
        el.appendChild(out);
        controls[def.key] = { out, def };
        break;
      }
      default:
        throw new Error(`panel: unknown control type '${def.type}'`);
    }
  }

  mount.appendChild(el);
  return {
    el,
    values,
    get: (key) => values[key],
    set(key, val) {
      const c = controls[key];
      if (!c?.input) return;
      c.input.value = val;
      c.input.dispatchEvent(new Event(c.def.type === 'select' ? 'change' : 'input'));
    },
    setLabel(key, text) {
      const c = controls[key];
      if (c?.input) c.input.textContent = text;
    },
    setReadout(key, text) {
      const c = controls[key];
      if (c?.out) c.out.textContent = text;
    },
  };
}
