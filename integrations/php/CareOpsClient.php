<?php

declare(strict_types=1);

function careopsRequest(string $path, ?array $payload = null): array
{
    $root = rtrim(getenv('CAREOPS_API_URL') ?: 'http://127.0.0.1:8000', '/');
    $handle = curl_init($root . $path);
    if ($handle === false) {
        throw new RuntimeException('Could not initialize HTTP client.');
    }
    curl_setopt_array($handle, [
        CURLOPT_RETURNTRANSFER => true,
        CURLOPT_TIMEOUT => $payload === null ? 5 : 60,
        CURLOPT_NOPROXY => '*',
        CURLOPT_HTTPHEADER => ['Accept: application/json', 'Content-Type: application/json'],
    ]);
    if ($payload !== null) {
        curl_setopt($handle, CURLOPT_POSTFIELDS, json_encode($payload, JSON_THROW_ON_ERROR));
    }
    $body = curl_exec($handle);
    $status = curl_getinfo($handle, CURLINFO_RESPONSE_CODE);
    if ($body === false || $status < 200 || $status >= 300) {
        throw new RuntimeException("CareOps API request failed (HTTP $status).");
    }
    $result = json_decode($body, true, 512, JSON_THROW_ON_ERROR);
    if (!is_array($result)) {
        throw new RuntimeException('Expected a JSON array or object.');
    }
    return $result;
}
