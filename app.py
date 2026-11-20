const STN = '143';       // 대구
const STN_NAME = '대구';
const BASE = 'https://apihub.kma.go.kr/api/typ01/url/';

// 단건 요청 (접속 실패 시 3회 재시도)
function fetchText_(url) {
  let lastErr;
  for (let a = 0; a < 3; a++) {
    try {
      return UrlFetchApp.fetch(url, { muteHttpExceptions: true }).getContentText('EUC-KR');
    } catch (e) {
      lastErr = e;
      Utilities.sleep(2000 * (a + 1));
    }
  }
  throw lastErr;
}

// 여러 건 요청 (실패하면 재시도 → 한 건씩 → 끝내 실패한 건은 null)
function fetchAllSafe_(reqs) {
  for (let a = 0; a < 3; a++) {
    try {
      return UrlFetchApp.fetchAll(reqs);
    } catch (e) {
      Utilities.sleep(2000 * (a + 1));
    }
  }
  return reqs.map(function (rq) {
    for (let a = 0; a < 2; a++) {
      try {
        return UrlFetchApp.fetch(rq.url, { muteHttpExceptions: true });
      } catch (e) {
        Utilities.sleep(1500);
      }
    }
    return null;
  });
}

// 결측 처리: 기온은 -99 이하만 결측(영하 9℃ 이하는 정상값), 습도는 음수면 결측
function tempVal_(s) { const v = parseFloat(s); return (isNaN(v) || v <= -99) ? '' : v; }
function humVal_(s)  { const v = parseFloat(s); return (isNaN(v) || v < 0) ? '' : v; }
// 강수량·적설: 숫자면 그대로, 음수(-9 등 '현상 없음' 표시)는 0, 빈 값/숫자 아님은 '' (미입력)
function precipVal_(s) { const v = parseFloat(s); if (isNaN(v)) return ''; return v < 0 ? 0 : v; }
function pickCol_(colIndex, names) {
  for (let i = 0; i < names.length; i++) {
    if (colIndex[names[i]] !== undefined) return colIndex[names[i]];
  }
  return undefined;
}

function addDays_(ymd, n) {
  const dt = new Date(Date.UTC(+ymd.slice(0, 4), +ymd.slice(4, 6) - 1, +ymd.slice(6, 8) + n));
  return Utilities.formatDate(dt, 'UTC', 'yyyyMMdd');
}

// ---------- 일자료 해석: [평균기온, 최저, 최고, 평균습도, 강수량(mm), 적설량(cm)] ----------
function parseDailyText_(text) {
  const colIndex = {};
  let row = null;
  text.split('\n').forEach(function (line) {
    const t = line.trim();
    if (!t) return;
    if (t.startsWith('#')) {
      const m = t.match(/^#\s*(\d+)\.\s*([A-Za-z0-9_]+)\s*:/);
      if (m) colIndex[m[2]] = Number(m[1]) - 1;
      return;
    }
    const p = t.split(/[\s,]+/);
    if (p[1] === STN) row = p;
  });
  if (!row || colIndex['TA_AVG'] === undefined) return null;
  if (colIndex['HM_AVG'] === undefined) {
    throw new Error('HM_AVG 항목을 찾지 못했습니다. 응답 항목: ' + Object.keys(colIndex).join(', '));
  }
  const rnIdx = pickCol_(colIndex, ['RN_DAY', 'RN']);       // 일 강수량(mm)
  const sdIdx = pickCol_(colIndex, ['SD_NEW', 'SD_DAY']);   // 일 최심신적설(cm) = 그날 새로 쌓인 눈
  if (rnIdx === undefined || sdIdx === undefined) {
    throw new Error('강수량(RN_DAY) 또는 신적설(SD_NEW) 항목을 찾지 못했습니다. 응답 항목: ' + Object.keys(colIndex).join(', '));
  }
  return [
    tempVal_(row[colIndex['TA_AVG']]),
    tempVal_(row[colIndex['TA_MIN']]),
    tempVal_(row[colIndex['TA_MAX']]),
    humVal_(row[colIndex['HM_AVG']]),
    precipVal_(row[rnIdx]),
    precipVal_(row[sdIdx])
  ];
}

function getDaily_(ymd) {
  const KEY = PropertiesService.getScriptProperties().getProperty('KMA_KEY');
  const text = fetchText_(BASE + 'kma_sfcdd.php?tm=' + ymd + '&stn=' + STN
                          + '&disp=1&help=1&authKey=' + KEY);
  const v = parseDailyText_(text);
  if (!v) throw new Error('일자료가 아직 없습니다. 응답 끝부분: ' + text.slice(-500));
  return v;
}

// ---------- 시간자료 해석 → { 'yyyyMMdd': [0시..23시] } ----------
function parseHourlyText_(text) {
  const colIndex = {};
  let headerCols = null;
  const rows = [];
  text.split('\n').forEach(function (line) {
    const t = line.trim();
    if (!t) return;
    if (t.startsWith('#')) {
      const m = t.match(/^#\s*(\d+)\.\s*([A-Za-z0-9_]+)\s*:/);
      if (m) { colIndex[m[2]] = Number(m[1]) - 1; return; }
      if (!t.includes(':')) {
        const toks = t.replace('#', '').trim().split(/\s+/);
        if (toks.indexOf('STN') >= 0 && toks.indexOf('TA') >= 0) headerCols = toks;
      }
      return;
    }
    const p = t.split(/[\s,]+/);
    if (/^\d{12}$/.test(p[0]) && p[1] === STN) rows.push(p);
  });
  let ti = colIndex['TA'];
  if (ti === undefined && headerCols) ti = headerCols.indexOf('TA');
  const out = {};
  if (ti === undefined || ti < 0) return out;
  rows.forEach(function (p) {
    const ymd = p[0].slice(0, 8);
    const hr = Number(p[0].slice(8, 10));
    if (!out[ymd]) out[ymd] = new Array(24).fill('');
    out[ymd][hr] = tempVal_(p[ti]);
  });
  return out;
}

function getHourly_(ymd) {
  const KEY = PropertiesService.getScriptProperties().getProperty('KMA_KEY');
  const text = fetchText_(BASE + 'kma_sfctm3.php?tm1=' + ymd + '0000&tm2=' + ymd
                          + '2300&stn=' + STN + '&help=1&authKey=' + KEY);
  const h = parseHourlyText_(text)[ymd];
  if (!h) throw new Error('시간자료가 아직 없습니다. 응답 끝부분: ' + text.slice(-500));
  return h;
}

// ---------- 시트 도우미 ----------
function findRow_(sheet, dateStr) {
  const dates = sheet.getRange(1, 1, sheet.getLastRow(), 1).getDisplayValues();
  for (let i = 0; i < dates.length; i++) {
    if (String(dates[i][0]).trim() === dateStr) return i + 1;
  }
  throw new Error('A열에서 ' + dateStr + ' 날짜 줄을 찾지 못했습니다.');
}

function hasBlank_(arr) {
  return arr.some(function (v) { return v === '' || v === null; });
}

// 비어 있는 칸에만 값을 채움 (이미 있는 값은 건드리지 않음)
function fillBlanks_(sheet, r, col, vals) {
  const rng = sheet.getRange(r, col, 1, vals.length);
  const cur = rng.getValues()[0];
  let changed = false;
  for (let i = 0; i < vals.length; i++) {
    if ((cur[i] === '' || cur[i] === null) && vals[i] !== '') {
      cur[i] = vals[i];
      changed = true;
    }
  }
  if (changed) rng.setValues([cur]);
}

// ================== ① 매일 자동 실행용 ==================
// 08:00, 08:30, 09:00, 09:30에 실행. 어제 줄(D~G, L~M, N~AL)이 모두 채워져 있으면 아무것도 하지 않음.
function fetchDaeguTemp() {
  const lock = LockService.getScriptLock();
  if (!lock.tryLock(30000)) return; // 동시에 두 번 실행되는 것 방지
  try {
    const d = new Date();
    d.setDate(d.getDate() - 1); // 어제
    const ymd = Utilities.formatDate(d, 'Asia/Seoul', 'yyyyMMdd');
    const dateStr = Utilities.formatDate(d, 'Asia/Seoul', 'yyyy-MM-dd');

    const sheet = SpreadsheetApp.getActiveSpreadsheet().getSheets()[0];
    const r = findRow_(sheet, dateStr);
    const msgs = [];

    const rngDG = function () { return sheet.getRange(r, 4, 1, 4).getValues()[0]; };    // D~G
    const rngLM = function () { return sheet.getRange(r, 12, 1, 2).getValues()[0]; };   // L~M
    const rngNAL = function () { return sheet.getRange(r, 14, 1, 25).getValues()[0]; }; // N~AL

    // D~G열: 평균기온 / 최저 / 최고 / 평균습도,  L열: 적설량(cm),  M열: 강수량(mm)
    const blankDG = hasBlank_(rngDG());
    const blankLM = hasBlank_(rngLM());
    if (blankDG || blankLM) {
      try {
        const dv = getDaily_(ymd);
        if (blankDG) fillBlanks_(sheet, r, 4, dv.slice(0, 4));
        if (blankLM) fillBlanks_(sheet, r, 12, [dv[5], dv[4]]);
      } catch (e) { msgs.push('일자료: ' + e.message); }
    }
    // N~AL열: 지점명, 0시~23시 기온
    if (hasBlank_(rngNAL())) {
      try { fillBlanks_(sheet, r, 14, [STN_NAME].concat(getHourly_(ymd))); }
      catch (e) { msgs.push('시간자료: ' + e.message); }
    }

    const incomplete = hasBlank_(rngDG()) || hasBlank_(rngLM()) || hasBlank_(rngNAL());
    if (!incomplete) { Logger.log(dateStr + ' 모두 채워짐'); return; }

    // 마지막 시도(09:20 이후)에도 빈 칸이 남으면 오류를 내서 알림이 가게 함
    const hhmm = Number(Utilities.formatDate(new Date(), 'Asia/Seoul', 'HHmm'));
    const msg = dateStr + ' 아직 빈 칸이 있습니다. ' + msgs.join(' | ');
    if (hhmm >= 920) throw new Error(msg);
    Logger.log(msg + ' (다음 실행에서 다시 시도)');
  } finally {
    lock.releaseLock();
  }
}

// ================== ② 자동 실행(트리거) 설정: 한 번만 실행 ==================
function setupTriggers() {
  ScriptApp.getProjectTriggers().forEach(function (t) {
    if (t.getHandlerFunction() === 'fetchDaeguTemp') ScriptApp.deleteTrigger(t);
  });
  [[8, 0], [8, 30], [9, 0], [9, 30]].forEach(function (hm) {
    ScriptApp.newTrigger('fetchDaeguTemp')
      .timeBased().everyDays(1).atHour(hm[0]).nearMinute(hm[1])
      .inTimezone('Asia/Seoul').create();
  });
  Logger.log('트리거 4개 생성 완료: 08:00, 08:30, 09:00, 09:30');
}

// ================== ②-2 강수/적설 열 준비: 한 번만 실행 ==================
// L1 = 적설량(cm), M1 = 강수량(mm) 머리글을 쓰고, 예전에 M열(지점)에 들어 있던 지점번호(143)를 지웁니다.
function setupPrecipColumns() {
  const sheet = SpreadsheetApp.getActiveSpreadsheet().getSheets()[0];
  sheet.getRange(1, 12).setValue('적설량(cm)');
  sheet.getRange(1, 13).setValue('강수량(mm)');
  const last = sheet.getLastRow();
  let cleared = 0;
  if (last >= 2) {
    const rng = sheet.getRange(2, 13, last - 1, 1);
    const vals = rng.getValues();
    for (let i = 0; i < vals.length; i++) {
      if (vals[i][0] !== '' && Number(vals[i][0]) === Number(STN)) { vals[i][0] = ''; cleared++; }
    }
    rng.setValues(vals);
  }
  Logger.log('머리글 설정 완료 / M열의 지점번호 ' + cleared + '칸 삭제');
}

// ================== ③ 확인용 (시트는 안 건드림) ==================
function testDailyLog() {
  Logger.log('11/12 [평균기온, 최저, 최고, 평균습도, 강수량, 적설량] = ' + getDaily_('20261112').join(', '));
}
// 비/눈이 왔던 것을 아는 날짜로 바꿔서 실행해 보세요. 강수량·적설량이 맞는지 확인용입니다.
function testPrecipLog() {
  ['20261105', '20261112', '20260115'].forEach(function (ymd) {
    Logger.log(ymd + ' [평균기온, 최저, 최고, 평균습도, 강수량(mm), 적설량(cm)] = ' + getDaily_(ymd).join(', '));
  });
}
// 일자료 응답에 어떤 항목(코드)이 있는지 보여줍니다. RN_DAY / SD_NEW 가 없다는 오류가 나면 여기서 확인하세요.
function listDailyItems() {
  const KEY = PropertiesService.getScriptProperties().getProperty('KMA_KEY');
  const d = new Date(); d.setDate(d.getDate() - 2);
  const ymd = Utilities.formatDate(d, 'Asia/Seoul', 'yyyyMMdd');
  const text = fetchText_(BASE + 'kma_sfcdd.php?tm=' + ymd + '&stn=' + STN + '&disp=1&help=1&authKey=' + KEY);
  text.split('\n').filter(function (l) { return l.trim().startsWith('#'); })
      .forEach(function (l) { Logger.log(l.trim()); });
}
function testHourlyLog() {
  Logger.log('10/31 시간별 기온: ' + getHourly_('20261031').join(', '));
}

// 응답 상태 확인용 (시트는 안 건드림, 인증키는 로그에 남기지 않음)
function debugDailyRaw() {
  const KEY = PropertiesService.getScriptProperties().getProperty('KMA_KEY');
  ['20261105', '20261118'].forEach(function (ymd) {
    const res = UrlFetchApp.fetch(BASE + 'kma_sfcdd.php?tm=' + ymd + '&stn=' + STN
                                  + '&disp=1&help=1&authKey=' + KEY, { muteHttpExceptions: true });
    const text = res.getContentText('EUC-KR');
    Logger.log(ymd + ' → HTTP ' + res.getResponseCode() + ' / 응답 길이 ' + text.length);
    Logger.log('앞부분: ' + text.slice(0, 300));
    Logger.log('끝부분: ' + text.slice(-300));
  });
}

// 기간 조회(kma_sfcdd3) 응답이 어떻게 생겼는지 확인용 (시트는 안 건드림, 인증키는 로그에 남기지 않음)
function testPeriodDaily() {
  const KEY = PropertiesService.getScriptProperties().getProperty('KMA_KEY');
  const res = UrlFetchApp.fetch(BASE + 'kma_sfcdd3.php?tm1=20160101&tm2=20160105&stn=' + STN
                                + '&help=1&authKey=' + KEY, { muteHttpExceptions: true });
  const text = res.getContentText('EUC-KR');
  Logger.log('HTTP ' + res.getResponseCode() + ' / 응답 길이 ' + text.length);
  Logger.log(text.slice(0, 1200));
}

// ================== ④ 강수량·적설량 과거 데이터 채우기 (자동 이어달리기) ==================
// backfillPrecip 을 한 번만 실행하면 됩니다.
// L열(적설량)·M열(강수량)이 비어 있는 날만 오래된 날짜부터 요청하고, 처리할 때마다 시트에 바로 저장합니다.
// 약 4분 반 일하고 멈춘 뒤, 남은 날이 있으면 1분 뒤 자동으로 다시 실행됩니다. 전부 끝나면 자동 실행도 해제됩니다.
// 도중에 멈추고 싶으면 stopPrecip 을 실행하세요.
function backfillPrecip() {
  const lock = LockService.getScriptLock();
  if (!lock.tryLock(20000)) { Logger.log('다른 작업이 실행 중이라 건너뜀'); return; }
  try {
    precipRun_();
  } finally {
    lock.releaseLock();
  }
}

function stopPrecip() {
  clearPrecipTriggers_();
  Logger.log('강수·적설 자동 이어달리기를 해제했습니다.');
}

function clearPrecipTriggers_() {
  ScriptApp.getProjectTriggers().forEach(function (t) {
    if (t.getHandlerFunction() === 'backfillPrecip') ScriptApp.deleteTrigger(t);
  });
}

// 일자료 한 건 재요청 (빈 응답일 때 사용). 결과: { v: 해석값 또는 null, info: 상태 설명 }
function retryDaily_(ymd) {
  const KEY = PropertiesService.getScriptProperties().getProperty('KMA_KEY');
  Utilities.sleep(1500);
  try {
    const res = UrlFetchApp.fetch(BASE + 'kma_sfcdd.php?tm=' + ymd + '&stn=' + STN
                                  + '&disp=1&help=1&authKey=' + KEY, { muteHttpExceptions: true });
    const text = res.getContentText('EUC-KR');
    return {
      v: parseDailyText_(text),
      info: 'HTTP ' + res.getResponseCode() + ' / 길이 ' + text.length + ' / 끝부분: ' + text.slice(-120).replace(/\n/g, ' ')
    };
  } catch (e) {
    return { v: null, info: '오류: ' + e.message };
  }
}

function precipRun_() {
  const t0 = Date.now();
  const BUDGET_MS = 270000; // 4분 30초
  const BATCH = 5;          // 한 번에 5건씩만 요청 (많이 보내면 기상청이 빈 응답을 줌)
  const KEY = PropertiesService.getScriptProperties().getProperty('KMA_KEY');
  const sheet = SpreadsheetApp.getActiveSpreadsheet().getSheets()[0];

  const yest = new Date();
  yest.setDate(yest.getDate() - 1);
  const yestYmd = Utilities.formatDate(yest, 'Asia/Seoul', 'yyyyMMdd');

  const aVals = sheet.getRange(1, 1, sheet.getLastRow(), 1).getDisplayValues();
  const rowOf = {};
  for (let i = 0; i < aVals.length; i++) {
    const s = String(aVals[i][0]).trim().replace(/-/g, '');
    if (/^\d{8}$/.test(s)) rowOf[s] = i + 1;
  }

  // 2016-01-01 ~ 어제 중 시트에 줄이 있는 날짜
  const all = [];
  for (let d = '20160101'; d <= yestYmd; d = addDays_(d, 1)) {
    if (rowOf[d]) all.push(d);
  }
  if (!all.length) { Logger.log('대상 날짜가 없습니다.'); clearPrecipTriggers_(); return; }

  const minRow = rowOf[all[0]];
  const maxRow = rowOf[all[all.length - 1]];
  const nRows = maxRow - minRow + 1;
  const block = sheet.getRange(minRow, 12, nRows, 2).getValues(); // L~M

  // L 또는 M이 비어 있는 날만 대상 (0은 값이 있는 것으로 봄)
  const todo = all.filter(function (d) {
    const v = block[rowOf[d] - minRow];
    return v[0] === '' || v[0] === null || v[1] === '' || v[1] === null;
  });
  if (!todo.length) {
    Logger.log('모든 날짜의 강수·적설이 채워져 있습니다. 완료!');
    clearPrecipTriggers_();
    return;
  }

  let done = 0, filled = 0, noData = 0, failed = 0, aborted = false;
  const blankDates = [];
  let noDataInfo = '';
  let sinceSave = 0;

  for (let i = 0; i < todo.length; i += BATCH) {
    if (Date.now() - t0 > BUDGET_MS) break;
    const part = todo.slice(i, i + BATCH);
    const reqs = part.map(function (d) {
      return {
        url: BASE + 'kma_sfcdd.php?tm=' + d + '&stn=' + STN + '&disp=1&help=1&authKey=' + KEY,
        muteHttpExceptions: true
      };
    });
    const res = fetchAllSafe_(reqs);
    for (let k = 0; k < part.length; k++) {
      const r = res[k];
      if (!r) { failed++; continue; }
      let v = null;
      try {
        v = parseDailyText_(r.getContentText('EUC-KR'));
      } catch (e) {
        Logger.log('해석 오류(' + part[k] + '): ' + e.message);
        aborted = true;
        break;
      }
      if (!v) {
        // 빈 응답이면 한 건씩 천천히 다시 요청
        const rr = retryDaily_(part[k]);
        v = rr.v;
        if (!v) {
          noData++;
          if (!noDataInfo) noDataInfo = part[k] + ' → ' + rr.info;
          continue;
        }
      }
      const idx = rowOf[part[k]] - minRow;
      if (v[5] !== '') block[idx][0] = v[5];  // L: 적설량
      if (v[4] !== '') block[idx][1] = v[4];  // M: 강수량
      if (v[5] === '' || v[4] === '') blankDates.push(part[k]);
      filled++;
    }
    if (aborted) break;
    done += part.length;
    sinceSave++;
    if (sinceSave >= 3) {
      sheet.getRange(minRow, 12, nRows, 2).setValues(block);
      sinceSave = 0;
    }
    Utilities.sleep(300);
  }
  sheet.getRange(minRow, 12, nRows, 2).setValues(block); // 마지막 저장

  const remain = todo.length - done;
  Logger.log('이번 실행: 요청 ' + done + '일 / 입력 ' + filled + '일 / 자료 없음 ' + noData
    + '일 / 접속 실패 ' + failed + '건 / 남은 날짜 약 ' + remain + '일 / 걸린 시간 '
    + Math.round((Date.now() - t0) / 1000) + '초');
  if (blankDates.length) {
    Logger.log('값이 비어 온 날짜(앞 10개): ' + blankDates.slice(0, 10).join(', '));
  }
  if (noDataInfo) {
    Logger.log('자료 없음 첫 사례: ' + noDataInfo);
  }

  clearPrecipTriggers_();
  if (aborted) {
    Logger.log('오류로 중단했습니다. 위 메시지를 확인하세요. (자동 재실행 안 함)');
  } else if (remain > 0 || failed > 0) {
    ScriptApp.newTrigger('backfillPrecip').timeBased().after(60 * 1000).create();
    Logger.log('1분 뒤 자동으로 이어서 실행됩니다.');
  } else {
    Logger.log('이번 범위 처리 완료. (값이 비어 온 날은 다음에 backfillPrecip 을 다시 실행하면 재시도합니다.)');
  }
}

// ================== ⑤ 시간별 기온 + D~G(기온·습도) 과거 데이터 채우기 (빈 곳 보정용) ==================
// ※ 강수량·적설량(L~M열)은 위의 backfillPrecip 이 담당합니다. 이쪽은 건드리지 않습니다.
function backfillTest() { backfill_('20160101', '20160103'); }
function backfill2016() { backfillYear_(2016); }
function backfill2017() { backfillYear_(2017); }
function backfill2018() { backfillYear_(2018); }
function backfill2019() { backfillYear_(2019); }
function backfill2020() { backfillYear_(2020); }
function backfill2021() { backfillYear_(2021); }
function backfill2022() { backfillYear_(2022); }
function backfill2023() { backfillYear_(2023); }
function backfill2024() { backfillYear_(2024); }
function backfill2025() { backfillYear_(2025); }
function backfill2026() { backfillYear_(2026); }

function backfillYear_(y) {
  const yest = new Date();
  yest.setDate(yest.getDate() - 1);
  const yestYmd = Utilities.formatDate(yest, 'Asia/Seoul', 'yyyyMMdd');
  let end = y + '1231';
  if (end > yestYmd) end = yestYmd;
  backfill_(y + '0101', end);
}

function backfill_(startYmd, endYmd) {
  const KEY = PropertiesService.getScriptProperties().getProperty('KMA_KEY');
  const sheet = SpreadsheetApp.getActiveSpreadsheet().getSheets()[0];

  const aVals = sheet.getRange(1, 1, sheet.getLastRow(), 1).getDisplayValues();
  const rowOf = {};
  for (let i = 0; i < aVals.length; i++) {
    const s = String(aVals[i][0]).trim().replace(/-/g, '');
    if (/^\d{8}$/.test(s)) rowOf[s] = i + 1;
  }

  const dates = [];
  for (let d = startYmd; d <= endYmd; d = addDays_(d, 1)) dates.push(d);
  dates.forEach(function (d) {
    if (!rowOf[d]) throw new Error('A열에 ' + d + ' 날짜 줄이 없습니다.');
  });
  const minRow = rowOf[dates[0]];
  const maxRow = rowOf[dates[dates.length - 1]];
  const nRows = maxRow - minRow + 1;
  let failed = 0;

  // 1) 시간별 기온 (30일 단위, 항상 덮어씀) → N~AL열(지점명 + 0~23시)
  const hReqs = [];
  for (let s = startYmd; s <= endYmd; s = addDays_(s, 30)) {
    let e = addDays_(s, 29);
    if (e > endYmd) e = endYmd;
    hReqs.push({
      url: BASE + 'kma_sfctm3.php?tm1=' + s + '0000&tm2=' + e + '2300&stn=' + STN
           + '&help=1&authKey=' + KEY,
      muteHttpExceptions: true
    });
  }
  const hourly = {};
  fetchAllSafe_(hReqs).forEach(function (r) {
    if (!r) { failed++; return; }
    const part = parseHourlyText_(r.getContentText('EUC-KR'));
    Object.keys(part).forEach(function (k) { hourly[k] = part[k]; });
  });

  const hBlock = sheet.getRange(minRow, 14, nRows, 25).getValues();
  let hCount = 0;
  dates.forEach(function (d) {
    if (hourly[d]) {
      hBlock[rowOf[d] - minRow] = [STN_NAME].concat(hourly[d]);
      hCount++;
    }
  });
  sheet.getRange(minRow, 14, nRows, 25).setValues(hBlock);

  // 2) 일자료: D~G열(평균기온/최저/최고/습도)에 빈 칸이 있는 날만 요청해서 빈 칸만 채움
  const dBlock = sheet.getRange(minRow, 4, nRows, 4).getValues();
  const need = dates.filter(function (d) {
    return hasBlank_(dBlock[rowOf[d] - minRow]);
  });

  const daily = {};
  const BATCH = 15;
  for (let i = 0; i < need.length; i += BATCH) {
    const part = need.slice(i, i + BATCH);
    const reqs = part.map(function (d) {
      return {
        url: BASE + 'kma_sfcdd.php?tm=' + d + '&stn=' + STN + '&disp=1&help=1&authKey=' + KEY,
        muteHttpExceptions: true
      };
    });
    fetchAllSafe_(reqs).forEach(function (r, k) {
      if (!r) { failed++; return; }
      const v = parseDailyText_(r.getContentText('EUC-KR'));
      if (v) daily[part[k]] = v;
    });
    Utilities.sleep(500);
  }

  let dCount = 0;
  need.forEach(function (d) {
    if (!daily[d]) return;
    const row = dBlock[rowOf[d] - minRow];
    for (let c = 0; c < 4; c++) {
      if ((row[c] === '' || row[c] === null) && daily[d][c] !== '') {
        row[c] = daily[d][c];
        dCount++;
      }
    }
  });
  sheet.getRange(minRow, 4, nRows, 4).setValues(dBlock);

  Logger.log('완료 ' + startYmd + '~' + endYmd + ' / 전체 ' + dates.length + '일'
    + ' / 시간별 기온 입력 ' + hCount + '일'
    + ' / D~G 빈 날 ' + need.length + '일 → 새로 채운 칸 ' + dCount + '개'
    + ' / 접속 실패 ' + failed + '건');
}
