// Run in an authenticated, disposable acceptance account with Playwright CLI:
// playwright-cli -s=rl4co-review run-code --filename scripts/verify_frontend_browser.js
async (page) => {
  const origin = new URL(page.url()).origin;
  const errors = [],
    checks = [],
    overflow = [];
  page.on('pageerror', (error) => errors.push(error.message));
  const routes = [
    ['config', '/'],
    ['benchmark', '/benchmark'],
    ['files', '/file_manager'],
    ['models', '/model_info'],
    ['model-detail', '/model_info/POMO'],
    ['profile', '/profile'],
  ];
  for (const width of [1440, 1024, 768, 390]) {
    await page.setViewportSize({ width, height: 1000 });
    for (const [name, route] of routes) {
      const response = await page.goto(origin + route);
      if (response.status() !== 200)
        throw new Error(route + ' HTTP ' + response.status());
      await page.waitForLoadState('networkidle');
      const layout = await page.evaluate(() => ({
        width: document.documentElement.clientWidth,
        scroll: document.documentElement.scrollWidth,
        offenders: [...document.querySelectorAll('main *')]
          .filter(
            (el) =>
              el.getBoundingClientRect().right > innerWidth + 2 &&
              el.getClientRects().length &&
              !el.closest('.table-scroll'),
          )
          .slice(0, 10)
          .map((el) => el.tagName + '.' + el.className),
      }));
      if (layout.scroll > layout.width + 1)
        overflow.push({ width, route, ...layout });
      if (width === 1440 || width === 390)
        await page.screenshot({
          path: 'output/playwright/' + name + '-' + width + '.png',
          fullPage: width === 390,
          animations: 'disabled',
        });
      checks.push({
        width,
        route,
        status: response.status(),
        overflow: layout.scroll - layout.width,
      });
    }
  }
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.goto(origin + '/');
  await page.waitForFunction(() => window.configurationValid === true);
  for (const [problem, group] of [
    ['tsp', null],
    ['atsp', null],
    ['mtsp', 'mtsp-params'],
    ['cvrp', 'cvrp-params'],
    ['sdvrp', 'cvrp-params'],
    ['vrptw', 'vrptw-params'],
    ['pdp', 'pdp-params'],
    ['op', 'op-params'],
    ['pctsp', 'pctsp-params'],
    ['spctsp', 'spctsp-params'],
    ['ffsp', 'ffsp-params'],
    ['chip_routing', 'chip-routing-params'],
  ]) {
    await page.locator('#problem-select').selectOption(problem);
    await page.waitForFunction(
      (problem) =>
        window.configurationValid &&
        document
          .getElementById('config-validation-alert')
          .textContent.includes(problem.toUpperCase()),
      problem,
    );
    if (group && !(await page.locator('#' + group).isVisible()))
      throw new Error('Missing parameters for ' + problem);
    checks.push({ problem, conditional: true, compatible: true });
  }
  await page.goto(origin + '/?model=pomo&problem=tsp');
  await page.waitForFunction(() => window.configurationValid);
  if ((await page.locator('#model-select').inputValue()) !== 'pomo')
    throw new Error('Model preselection lost');
  if (!(await page.locator('#pomo-params').isVisible()))
    throw new Error('POMO parameters missing');
  await page.locator('[data-open-assistant]').click();
  await page.screenshot({ path: 'output/playwright/assistant-1440.png', animations: 'disabled' });
  await page.keyboard.press('Escape');
  if (await page.locator('#assistant-layer').isVisible())
    throw new Error('Drawer Escape failed');
  await page.setViewportSize({ width: 390, height: 844 });
  await page.locator('[data-open-assistant]').click();
  await page.screenshot({ path: 'output/playwright/assistant-390.png', animations: 'disabled' });
  await page.keyboard.press('Escape');
  await page.setViewportSize({ width: 1024, height: 900 });
  await page.locator('.menu-toggle').click();
  await page.keyboard.press('Escape');
  await page.setViewportSize({ width: 1440, height: 1000 });
  if (!(await page.locator('#app-sidebar').isVisible()))
    throw new Error('Sidebar disappeared after resizing');
  if (overflow.length || errors.length)
    throw new Error('Browser verification found issues');
  return { checks, overflow, pageErrors: errors };
}
