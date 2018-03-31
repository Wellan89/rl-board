
class Referee extends MultiReferee {

    private ActionResult computeMove(Unit unit, String dir1, String dir2) throws LostException {

        Point target = getNeighbor(dir1, unit.position);
        Integer targetHeight = grid.get(target);
        if (targetHeight == null) {
            throw new LostException("BadCoords", target.x, target.y);
        }
        int currentHeight = grid.get(unit.position);
        if (targetHeight > currentHeight + 1) {
            throw new LostException("InvalidMove", currentHeight, targetHeight);
        }
        if (targetHeight >= FINAL_HEIGHT) {
            throw new LostException("MoveTooHigh", target.x, target.y);
        }
        if (getUnitOnPoint(target).isPresent()) {
            throw new LostException("MoveOnUnit", target.x, target.y);
        }

        Point placeTarget = getNeighbor(dir2, target);
        Integer placeTargetHeight = grid.get(placeTarget);
        if (placeTargetHeight == null) {
            throw new LostException("InvalidPlace", placeTarget.x, placeTarget.y);
        }
        if (placeTargetHeight >= FINAL_HEIGHT) {
            throw new LostException("PlaceTooHigh", targetHeight);
        }

        ActionResult result = new ActionResult(Action.MOVE);
        result.moveTarget = target;
        result.placeTarget = placeTarget;

        Optional<Unit> possibleUnit = getUnitOnPoint(placeTarget).filter(u -> !u.equals(unit));
        if (!possibleUnit.isPresent()) {
            result.placeValid = true;
            result.moveValid = true;
        } else if (FOG_OF_WAR && !unitVisibleToPlayer(possibleUnit.get(), unit.player)) {
            result.placeValid = false;
            result.moveValid = true;
        } else {
            throw new LostException("PlaceOnUnit", placeTarget.x, placeTarget.y);
        }

        if (targetHeight == FINAL_HEIGHT - 1) {
            result.scorePoint = true;
        }
        result.unit = unit;
        return result;
    }

    private ActionResult computePush(Unit unit, String dir1, String dir2) throws LostException {
        if (!validPushDirection(dir1, dir2)) {
            throw new LostException("PushInvalid", dir1, dir2);
        }
        Point target = getNeighbor(dir1, unit.position);
        Optional<Unit> maybePushed = getUnitOnPoint(target);
        if (!maybePushed.isPresent()) {
            throw new LostException("PushVoid", target.x, target.y);
        }
        Unit pushed = maybePushed.get();

        if (pushed.player == unit.player) {
            throw new LostException("FriendlyFire", unit.index, pushed.index);
        }

        Point pushTo = getNeighbor(dir2, pushed.position);
        Integer toHeight = grid.get(pushTo);
        int fromHeight = grid.get(target);

        if (toHeight == null || toHeight >= FINAL_HEIGHT || toHeight > fromHeight + 1) {
            throw new LostException("PushInvalid", dir1, dir2);
        }

        ActionResult result = new ActionResult(Action.PUSH);
        result.moveTarget = pushTo;
        result.placeTarget = target;

        Optional<Unit> possibleUnit = getUnitOnPoint(pushTo);
        if (!possibleUnit.isPresent()) {
            result.placeValid = true;
            result.moveValid = true;
        } else if (FOG_OF_WAR && !unitVisibleToPlayer(possibleUnit.get(), unit.player)) {
            result.placeValid = false;
            result.moveValid = false;

        } else {
            throw new LostException("PushOnUnit", dir1, dir2);
        }

        result.unit = pushed;

        return result;
    }

    @Override
    protected void handlePlayerOutput(int frame, int round, int playerIdx, String[] outputs)
            throws WinException, LostException, InvalidInputException {
        String line = outputs[0];
        Player player = players.get(playerIdx);

        try {
            Matcher match = ACCEPT_DEFEAT_PATTERN.matcher(line);
            if (match.matches()) {
                player.die(round);
                //Message
                matchMessage(player, match);
                throw new LostException("selfDestruct", player.index);
            }
            match = PLAYER_PATTERN.matcher(line);
            if (match.matches()) {
                String action = match.group("action");
                String indexString = match.group("index");
                String dir1 = match.group("move").toUpperCase();
                String dir2 = match.group("place").toUpperCase();
                int index = Integer.valueOf(indexString);
                Unit unit = player.units.get(index);

                ActionResult ar = computeAction(action, unit, dir1, dir2);
                unit.did = ar;
                if (ar.moveValid) {
                    ar.unit.position = ar.moveTarget;
                }
                if (ar.placeValid) {
                    grid.place(ar.placeTarget);
                }
                if (ar.scorePoint) {
                    player.score++;
                }
                if (ar.type.equals(Action.PUSH)) {
                    ar.unit.gotPushed = true;
                }

                //Message
                matchMessage(player, match);
                return;
            }

            throw new InvalidInputException(expected, line);

        } catch (LostException | InvalidInputException e) {
            player.die(round);
            throw e;
        } catch (Exception e) {
            StringWriter errors = new StringWriter();
            e.printStackTrace(new PrintWriter(errors));
            printError(e.getMessage() + "\n" + errors.toString());
            player.die(round);
            throw new InvalidInputException(expected, line);
        }

    }


}
