<?php

declare(strict_types=1);

use Illuminate\Foundation\Application;
use Illuminate\Foundation\Configuration\Exceptions;
use Illuminate\Support\Facades\Route;

require dirname(__DIR__, 2) . '/CareOpsClient.php';

return Application::configure(basePath: dirname(__DIR__))
    ->withRouting(using: static function (): void {
        Route::get('/summary', static function () {
            try {
                $rows = careopsRequest('/runs?limit=10');
                if (!array_is_list($rows)) {
                    throw new RuntimeException('Expected an array of run summaries.');
                }
                $outcomes = [];
                foreach ($rows as $row) {
                    $name = $row['status'] ?? null;
                    if (!is_string($name)) {
                        throw new RuntimeException('Expected a status string for each run.');
                    }
                    $outcomes[$name] = ($outcomes[$name] ?? 0) + 1;
                }
                return response()->json(['runs' => count($rows), 'outcomes' => (object) $outcomes]);
            } catch (RuntimeException | JsonException) {
                return response()->json(['detail' => 'CareOps API unavailable'], 502);
            }
        });
    })
    ->withExceptions(static function (Exceptions $exceptions): void {
        $exceptions->shouldRenderJsonWhen(static fn (): bool => true);
    })
    ->create();
