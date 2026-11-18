<?php

declare(strict_types=1);

require __DIR__ . '/CareOpsClient.php';

if ($argc !== 3 || !preg_match('/^SYN-D-[0-9]{3}$/D', $argv[1])
    || !in_array($argv[2], ['--allow', '--decline'], true)) {
    fwrite(STDERR, "Usage: php review.php SYN-D-001 --allow|--decline\n");
    exit(2);
}

try {
    $pending = careopsRequest('/reviews', ['claim_id' => $argv[1]]);
    $run = $pending['run_id'] ?? '';
    if (($pending['status'] ?? null) !== 'awaiting_approval' || !preg_match('/^[0-9a-f]{32}$/D', $run)) {
        throw new RuntimeException('Expected a pending review.');
    }
    $result = careopsRequest("/reviews/$run/decision", ['approved' => $argv[2] === '--allow']);
    if (($result['run_id'] ?? null) !== $run || ($result['status'] ?? null) !== 'complete') {
        throw new RuntimeException('Expected a completed review.');
    }
    echo json_encode($result, JSON_THROW_ON_ERROR | JSON_PRETTY_PRINT), PHP_EOL;
} catch (RuntimeException | JsonException $error) {
    fwrite(STDERR, $error->getMessage() . PHP_EOL);
    exit(1);
}
