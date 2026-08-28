/*
 * Interactive scale-bar canvas.
 *
 * Gradio has no component for "drag a rectangle" or "click a point precisely",
 * so this draws the micrograph on a canvas and handles the interaction itself.
 * Results are pushed back into hidden Gradio textboxes, which is the supported
 * way for client code to hand values to a Python callback.
 *
 * Two modes:
 *   box    - a rectangle with corner handles; drag a corner to resize, drag the
 *            middle to move. Used to tell OCR where the scale bar is.
 *   points - click each end of the bar. A magnifier follows the cursor so the
 *            end of a bar can be hit exactly rather than approximately.
 */
(function () {
  "use strict";

  const HANDLE = 9;          // corner grab square, display px
  const LOUPE = 150;         // magnifier box, display px
  const LOUPE_ZOOM = 7;
  const EDGE_PAD = 14;

  const S = {
    img: null, iw: 0, ih: 0,
    mode: "box",
    box: null,               // {x0,y0,x1,y1} in IMAGE coords
    points: [],              // [[x,y], ...] in IMAGE coords
    drag: null,              // {kind, corner, grabDX, grabDY}
    cursor: null,            // [x,y] image coords, for the loupe
    view: { s: 1, ox: 0, oy: 0 },
  };

  const $ = (sel) => document.querySelector(sel);

  function canvas() { return $("#scale_canvas"); }

  /* ---------- Gradio interop ---------- */

  function pushTo(elemId, payload) {
    const ta = document.querySelector("#" + elemId + " textarea");
    if (!ta) return false;
    const setter = Object.getOwnPropertyDescriptor(
      window.HTMLTextAreaElement.prototype, "value").set;
    setter.call(ta, JSON.stringify(payload));
    ta.dispatchEvent(new Event("input", { bubbles: true }));
    return true;
  }

  function readChannel(elemId) {
    const ta = document.querySelector("#" + elemId + " textarea");
    if (!ta || !ta.value) return null;
    try { return JSON.parse(ta.value); } catch (e) { return null; }
  }

  /* ---------- coordinate mapping ---------- */

  function fitView() {
    const c = canvas();
    if (!c || !S.iw) return;
    const wrap = c.parentElement;
    const maxW = Math.max(320, wrap.clientWidth - 4);
    const maxH = 620;
    const s = Math.min(maxW / S.iw, maxH / S.ih, 1);
    c.width = Math.round(S.iw * s);
    c.height = Math.round(S.ih * s);
    S.view = { s: s, ox: 0, oy: 0 };
  }

  const toDisplay = (x, y) => [x * S.view.s, y * S.view.s];
  const toImage = (x, y) => [x / S.view.s, y / S.view.s];

  function eventImageCoords(ev) {
    const c = canvas();
    const r = c.getBoundingClientRect();
    // A hidden tab has a zero-size rect; scaling by it yields NaN coordinates
    // that would be committed as a garbage box.
    if (!r.width || !r.height || !S.view.s) return null;
    // The canvas may be laid out at a different size than its backing store.
    const bx = (ev.clientX - r.left) * (c.width / r.width);
    const by = (ev.clientY - r.top) * (c.height / r.height);
    const [ix, iy] = toImage(bx, by);
    return [clamp(ix, 0, S.iw), clamp(iy, 0, S.ih)];
  }

  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

  /* ---------- drawing ---------- */

  function draw() {
    const c = canvas();
    if (!c || !S.img) return;
    const ctx = c.getContext("2d");
    ctx.clearRect(0, 0, c.width, c.height);
    ctx.drawImage(S.img, 0, 0, c.width, c.height);

    if (S.mode === "box" && S.box) drawBox(ctx);
    if (S.mode === "points") drawPoints(ctx);
    if (S.mode === "points" && S.cursor) drawLoupe(ctx);
  }

  function drawBox(ctx) {
    const [x0, y0] = toDisplay(S.box.x0, S.box.y0);
    const [x1, y1] = toDisplay(S.box.x1, S.box.y1);

    ctx.save();
    // Dim everything outside the region so the search area is unmistakable.
    ctx.fillStyle = "rgba(0,0,0,0.45)";
    ctx.beginPath();
    ctx.rect(0, 0, ctx.canvas.width, ctx.canvas.height);
    ctx.rect(x0, y0, x1 - x0, y1 - y0);
    ctx.fill("evenodd");

    ctx.strokeStyle = "#22d3ee";
    ctx.lineWidth = 2;
    ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);

    ctx.fillStyle = "#22d3ee";
    for (const [hx, hy] of corners(x0, y0, x1, y1)) {
      ctx.fillRect(hx - HANDLE / 2, hy - HANDLE / 2, HANDLE, HANDLE);
    }
    ctx.restore();
  }

  const corners = (x0, y0, x1, y1) =>
    [[x0, y0], [x1, y0], [x0, y1], [x1, y1]];

  function drawPoints(ctx) {
    if (S.points.length === 2) {
      const [a, b] = S.points.map(([x, y]) => toDisplay(x, y));
      ctx.save();
      ctx.strokeStyle = "#f97316";
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 4]);
      ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
      ctx.restore();
    }
    S.points.forEach(([ix, iy], i) => {
      const [x, y] = toDisplay(ix, iy);
      ctx.save();
      ctx.strokeStyle = "#f97316";
      ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(x, y, 7, 0, Math.PI * 2); ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(x - 11, y); ctx.lineTo(x + 11, y);
      ctx.moveTo(x, y - 11); ctx.lineTo(x, y + 11);
      ctx.stroke();
      ctx.fillStyle = "#f97316";
      ctx.font = "bold 13px system-ui, sans-serif";
      ctx.fillText(String(i + 1), x + 12, y - 10);
      ctx.restore();
    });
  }

  function drawLoupe(ctx) {
    const c = ctx.canvas;
    const [ix, iy] = S.cursor;
    const [cx, cy] = toDisplay(ix, iy);

    // Keep the loupe clear of the cursor and inside the canvas.
    let lx = cx + 24, ly = cy + 24;
    if (lx + LOUPE > c.width - EDGE_PAD) lx = cx - LOUPE - 24;
    if (ly + LOUPE > c.height - EDGE_PAD) ly = cy - LOUPE - 24;
    lx = clamp(lx, EDGE_PAD, c.width - LOUPE - EDGE_PAD);
    ly = clamp(ly, EDGE_PAD, c.height - LOUPE - EDGE_PAD);

    // Source window, in the ORIGINAL image, so the zoom shows real pixels
    // rather than an interpolation of the already-downscaled canvas.
    const src = LOUPE / LOUPE_ZOOM;
    const sx = ix - src / 2, sy = iy - src / 2;

    ctx.save();
    ctx.beginPath();
    ctx.rect(lx, ly, LOUPE, LOUPE);
    ctx.clip();
    ctx.fillStyle = "#000";
    ctx.fillRect(lx, ly, LOUPE, LOUPE);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(S.img, sx, sy, src, src, lx, ly, LOUPE, LOUPE);
    ctx.restore();

    ctx.save();
    ctx.strokeStyle = "#f97316";
    ctx.lineWidth = 2;
    ctx.strokeRect(lx, ly, LOUPE, LOUPE);
    // Crosshair marks the exact pixel that a click would land on.
    const mx = lx + LOUPE / 2, my = ly + LOUPE / 2;
    ctx.beginPath();
    ctx.moveTo(mx - 14, my); ctx.lineTo(mx - 3, my);
    ctx.moveTo(mx + 3, my); ctx.lineTo(mx + 14, my);
    ctx.moveTo(mx, my - 14); ctx.lineTo(mx, my - 3);
    ctx.moveTo(mx, my + 3); ctx.lineTo(mx, my + 14);
    ctx.stroke();

    ctx.fillStyle = "rgba(0,0,0,0.75)";
    ctx.fillRect(lx, ly + LOUPE - 20, LOUPE, 20);
    ctx.fillStyle = "#fff";
    ctx.font = "12px ui-monospace, monospace";
    ctx.fillText(`x ${Math.round(ix)}  y ${Math.round(iy)}`, lx + 6, ly + LOUPE - 6);
    ctx.restore();
  }

  /* ---------- interaction ---------- */

  function hitCorner(ix, iy) {
    if (!S.box) return null;
    const [dx, dy] = toDisplay(ix, iy);
    const [x0, y0] = toDisplay(S.box.x0, S.box.y0);
    const [x1, y1] = toDisplay(S.box.x1, S.box.y1);
    const names = ["nw", "ne", "sw", "se"];
    const pts = corners(x0, y0, x1, y1);
    for (let i = 0; i < 4; i++) {
      if (Math.abs(dx - pts[i][0]) <= HANDLE && Math.abs(dy - pts[i][1]) <= HANDLE) {
        return names[i];
      }
    }
    return null;
  }

  const inBox = (ix, iy) => S.box &&
    ix > S.box.x0 && ix < S.box.x1 && iy > S.box.y0 && iy < S.box.y1;

  function onDown(ev) {
    if (S.mode !== "box" || !S.img) return;
    const at = eventImageCoords(ev);
    if (!at) return;
    const [ix, iy] = at;
    const corner = hitCorner(ix, iy);
    if (corner) {
      S.drag = { kind: "resize", corner };
    } else if (inBox(ix, iy)) {
      S.drag = { kind: "move", gx: ix - S.box.x0, gy: iy - S.box.y0 };
    } else {
      S.box = { x0: ix, y0: iy, x1: ix, y1: iy };
      S.drag = { kind: "resize", corner: "se" };
    }
    ev.preventDefault();
    draw();
  }

  function onMove(ev) {
    if (!S.img) return;
    const at = eventImageCoords(ev);
    if (!at) return;
    const [ix, iy] = at;
    S.cursor = [ix, iy];

    if (S.mode === "box") {
      canvas().style.cursor =
        hitCorner(ix, iy) ? "nwse-resize" : (inBox(ix, iy) ? "move" : "crosshair");
      if (S.drag) {
        if (S.drag.kind === "move") {
          const w = S.box.x1 - S.box.x0, h = S.box.y1 - S.box.y0;
          S.box.x0 = clamp(ix - S.drag.gx, 0, S.iw - w);
          S.box.y0 = clamp(iy - S.drag.gy, 0, S.ih - h);
          S.box.x1 = S.box.x0 + w;
          S.box.y1 = S.box.y0 + h;
        } else {
          const c = S.drag.corner;
          if (c.includes("w")) S.box.x0 = ix; else S.box.x1 = ix;
          if (c.includes("n")) S.box.y0 = iy; else S.box.y1 = iy;
        }
      }
    } else {
      canvas().style.cursor = "none";  // the loupe crosshair is the cursor
    }
    draw();
  }

  function onUp() {
    if (S.mode === "box" && S.drag) {
      S.drag = null;
      normaliseBox();
      commitBox();
      draw();
    }
  }

  function onLeave() { S.cursor = null; draw(); }

  function onClick(ev) {
    if (S.mode !== "points" || !S.img) return;
    const at = eventImageCoords(ev);
    if (!at) return;
    const [ix, iy] = at;
    if (S.points.length >= 2) S.points = [];
    S.points.push([ix, iy]);
    commitPoints();
    draw();
  }

  function normaliseBox() {
    const b = S.box;
    S.box = {
      x0: clamp(Math.min(b.x0, b.x1), 0, S.iw),
      y0: clamp(Math.min(b.y0, b.y1), 0, S.ih),
      x1: clamp(Math.max(b.x0, b.x1), 0, S.iw),
      y1: clamp(Math.max(b.y0, b.y1), 0, S.ih),
    };
  }

  const commitBox = () => S.box && pushTo("scale_box_out", {
    x0: Math.round(S.box.x0), y0: Math.round(S.box.y0),
    x1: Math.round(S.box.x1), y1: Math.round(S.box.y1),
  });

  const commitPoints = () => pushTo("scale_points_out", {
    points: S.points.map(([x, y]) => [+x.toFixed(2), +y.toFixed(2)]),
  });

  /* ---------- public API, called from Gradio events ---------- */

  function defaultBox() {
    // Most instruments print the bar along the bottom; start there so the box
    // usually needs a nudge rather than being drawn from scratch.
    return {
      x0: Math.round(S.iw * 0.02), y0: Math.round(S.ih * 0.88),
      x1: Math.round(S.iw * 0.45), y1: Math.round(S.ih * 0.99),
    };
  }

  window.SCALE = {
    load: function () {
      const data = readChannel("scale_image_in");
      if (!data || !data.img) return;
      const img = new Image();
      img.onload = function () {
        S.img = img; S.iw = img.naturalWidth; S.ih = img.naturalHeight;
        S.points = [];
        // Python sends the region automatic detection used, when it found one,
        // so the analyst sees what was measured instead of a generic guess.
        S.box = data.box ? {
          x0: clamp(data.box.x0, 0, S.iw), y0: clamp(data.box.y0, 0, S.ih),
          x1: clamp(data.box.x1, 0, S.iw), y1: clamp(data.box.y1, 0, S.ih),
        } : defaultBox();
        fitView();
        commitBox();
        draw();
      };
      img.src = data.img;
    },
    setMode: function (mode) {
      S.mode = mode === "points" ? "points" : "box";
      S.cursor = null;
      draw();
    },
    reset: function () {
      S.points = [];
      S.box = S.img ? defaultBox() : null;
      commitBox(); commitPoints(); draw();
    },
    attach: function () {
      const c = canvas();
      if (!c || c.dataset.wired) return;
      c.dataset.wired = "1";
      c.addEventListener("mousedown", onDown);
      c.addEventListener("mousemove", onMove);
      window.addEventListener("mouseup", onUp);
      c.addEventListener("mouseleave", onLeave);
      c.addEventListener("click", onClick);
      window.addEventListener("resize", function () { fitView(); draw(); });
      // The canvas has no size until its tab is shown, so re-fit when it does.
      if (window.ResizeObserver) {
        new ResizeObserver(function () {
          if (S.img && canvas().getBoundingClientRect().width) { fitView(); draw(); }
        }).observe(c.parentElement);
      }
    },
  };

  // The canvas arrives with the Gradio app, so wait for it.
  const poll = setInterval(function () {
    if (canvas()) { window.SCALE.attach(); clearInterval(poll); }
  }, 200);
})();
